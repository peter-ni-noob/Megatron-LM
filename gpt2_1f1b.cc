#include <cnpy.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <nvtx3/nvtx3.hpp>
#include <random>
#include <sstream>

#include "nexus/NexusCpp.hpp"
#include "nexus/common/base.hpp"
#include "nexus/distributed/distributed.hpp"
#include "nexus/distributed/parallel_state.hpp"
#include "nexus/distributed/pipeline_parallel/pipeline_schedule.hpp"
#include "nexus/operator/fcompute_gpu.hpp"

std::string test_data_path{"/data/lambada/"};
std::string train_data_path{"/data/dataset_1b/"};
std::string load_weight_path{"/data/Pweight/P_weightXL0/"};
std::string save_check_path{"/data/Pweight/N_weight15/"};

namespace cnpy {
void print_shape(const std::vector<size_t>& shape) {
  printf("[");
  for (int64_t i = 0; i < static_cast<int64_t>(shape.size()) - 1; i++)
    printf("%lu, ", shape[i]);
  if (shape.size() > 0) printf("%lu", shape.back());
  printf("]\n");
}

// Load into an existing NpyArray, rather than creating a new one.
// Can avoid allocating memory, which may save some time.
void load_the_npy_file(NpyArray* out, FILE* fp) {
  std::vector<size_t> shape;
  size_t word_size;
  bool fortran_order;
  cnpy::parse_npy_header(fp, word_size, shape, fortran_order);

  // check shape, word_size and fortran_order
  if (shape != out->shape || word_size != out->word_size ||
      fortran_order != out->fortran_order) {
    printf("Existing meta data:\n");
    printf("  shape: ");
    print_shape(out->shape);
    printf("  word_size: %lu\n", out->word_size);
    printf("  fortran_order: %d\n", out->fortran_order);

    printf("New meta data:\n");
    printf("  shape: ");
    print_shape(shape);
    printf("  word_size: %lu\n", word_size);
    printf("  fortran_order: %d\n", fortran_order);

    throw std::runtime_error("load_the_npy_file: meta data not the same.");
  }
  size_t nread = fread(out->data<char>(), 1, out->num_bytes(), fp);
  if (nread != out->num_bytes())
    throw std::runtime_error("load_the_npy_file: failed fread");
}

// Load into an existing NpyArray, rather than creating a new one.
// Can avoid allocating memory, which may save some time.
void npy_load(NpyArray* out, std::string fname) {
  FILE* fp = fopen(fname.c_str(), "rb");

  if (!fp) throw std::runtime_error("npy_load: Unable to open file " + fname);

  load_the_npy_file(out, fp);

  fclose(fp);
}
}  // namespace cnpy

// dataloader
class GPT2_Data_Loader {
 public:
  GPT2_Data_Loader() { init_load_queue(); }

  // 用户初始化
  void Init(int batchsize, Context ctx, int token_len = 1024,
            bool shuffle = false) {
    batch_size_ = batchsize;
    ctx_ = ctx;
    token_length = token_len;
    shuffle_ = shuffle;
    CUDA_CHECK(cudaMallocHost(&cpu_batch_buffer_,
                              sizeof(float) * batchsize * token_len));
    CUDA_CHECK(cudaMallocHost(&label_cpu_batch_buffer_,
                              sizeof(float) * batchsize * token_len));
    if (distributed::get_rank() == 0) {
      printf(
          "[GPT2 DataLoader] Batch size: %d, token length: %d, total tokens: "
          "%lld, "
          "shuffle: ",
          batch_size_, token_length, (long long)total_data_size * token_length);
      std::cout << std::boolalpha << shuffle_ << std::endl;
    }
  }

  ~GPT2_Data_Loader() {
    CUDA_CHECK(cudaFreeHost(cpu_batch_buffer_));
    CUDA_CHECK(cudaFreeHost(label_cpu_batch_buffer_));
  }

  void BeforeFirst() {
    if (shuffle_) shuffle_load_queue();
    point_now = 0;
    point_next = point_now;
  }

  bool Next() {
    if (point_next == 0) {
      point_next += this->batch_size_;
      if (point_next > total_data_size) {
        point_next = total_data_size;
      }
      return true;
    }
    point_next += this->batch_size_;
    if (point_now >= total_data_size) {
      return false;
    }
    if (point_next > total_data_size) {
      point_next = total_data_size;
    }
    return true;
  }

  io::TensorBatch Value() {
    io::TensorBatch tb;
    tb.data.emplace_back(TShape{(size_t)batch_size_, (size_t)token_length},
                         kFloat32, ctx_);
    tb.data.emplace_back(TShape{(size_t)batch_size_, (size_t)token_length},
                         kFloat32, ctx_);

    int real_batch_size = point_next - point_now;

    for (int i = 0; i < real_batch_size; i++) {
      int dir = now_data_left_copy[point_now] / 10000;
      int idx = now_data_left_copy[point_now] % 10000;
      point_now++;

      cnpy::NpyArray* arr = &arr_10000_;
      cnpy::NpyArray* label_arr = &label_arr_10000;
      if (dir == 888) {
        arr = &arr_1886_;
        label_arr = &label_arr_1886;
      }

      // If we need to read a new npy file
      if (dir != last_dir_) {
        // printf("%d\n", dir);
        std::filesystem::path directory_name_p{std::to_string(dir)};
        std::filesystem::path file_name{"token_float32.npy"};
        std::filesystem::path real_path{main_directory_path / directory_name_p /
                                        file_name};
        std::filesystem::path label_file_name{"label_float32.npy"};
        std::filesystem::path label_real_path{
            main_directory_path / directory_name_p / label_file_name};

        // Load into existing NdArray if possible
        if (arr->num_vals == 0) {
          *arr = cnpy::npy_load(real_path);
        } else {
          cnpy::npy_load(arr, real_path);
        }
        if (label_arr->num_vals == 0) {
          *label_arr = cnpy::npy_load(label_real_path);
        } else {
          cnpy::npy_load(label_arr, label_real_path);
        }
      }
      last_dir_ = dir;

      float* src_ptr = arr->data<float>();
      float* label_src_ptr = label_arr->data<float>();
      memcpy(cpu_batch_buffer_ + i * token_length, src_ptr + idx * token_length,
             sizeof(float) * token_length);
      memcpy(label_cpu_batch_buffer_ + i * token_length,
             label_src_ptr + idx * token_length, sizeof(float) * token_length);
    }

    // Padding: repeat the last sequence
    for (int i = real_batch_size; i < batch_size_; i++) {
      memcpy(cpu_batch_buffer_ + i * token_length,
             cpu_batch_buffer_ + real_batch_size * token_length,
             sizeof(float) * token_length);
      memcpy(label_cpu_batch_buffer_ + i * token_length,
             label_cpu_batch_buffer_ + real_batch_size * token_length,
             sizeof(float) * token_length);
    }

    // Copy the whole batch from CPU (page-locked memory) to GPU global memory
    float* dev_ptr = tb.data[0].GetData().dptr<float>();
    float* label_dev_ptr = tb.data[1].GetData().dptr<float>();
    CUDA_CHECK(cudaMemcpy(dev_ptr, cpu_batch_buffer_,
                          sizeof(float) * batch_size_ * token_length,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(label_dev_ptr, label_cpu_batch_buffer_,
                          sizeof(float) * batch_size_ * token_length,
                          cudaMemcpyHostToDevice));

    return tb;
  }

  void init_load_queue() {
    now_data_left.reserve(total_data_size);
    for (int i = 0; i < total_data_size; i++) {
      now_data_left.push_back(i);
    }
    now_data_left_copy = now_data_left;
  }

  void shuffle_load_queue() {
    now_data_left_copy.clear();
    now_data_left_copy = now_data_left;
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(now_data_left_copy.begin(), now_data_left_copy.end(), rng);
  }

 private:
  std::filesystem::path main_directory_path{
      train_data_path};  //"/data/dataset_1b"
  //   std::filesystem::path
  //   main_directory_path{"/data/openwebtext-npy/dataset_1b"};
  // Remember the last npy file and reuse it when possible
  int last_dir_ = -1;

  // Always load into these two NpyArray (avoid creating new ones)
  cnpy::NpyArray arr_10000_, arr_1886_, label_arr_10000, label_arr_1886;

  // Don't cudaMemcpy for each sample (too much overhead), cudaMemcpy the whole
  // batch
  float* cpu_batch_buffer_;
  float* label_cpu_batch_buffer_;

  // Whether to shuffle on each BeforeFirst()
  bool shuffle_;

  std::vector<int> now_data_left;
  std::vector<int> now_data_left_copy;
  int total_data_size{100 * 10000};
  int token_length{1024};
  int batch_size_{32};
  int point_now{0};
  int point_next{0};
  Context ctx_;
};

class Anneals_Cosine_LR_Scheduler {
 public:
  Anneals_Cosine_LR_Scheduler(double init_lr = 1e-4,
                              double max_lr = 1.5e-3,  // 0.0006
                              double min_lr = 1e-4, int lr_warmup_steps = 700,
                              int lr_decay_steps = 32000)
      : init_lr_(init_lr),
        max_lr_(max_lr),
        min_lr_(min_lr),
        lr_warmup_steps_(lr_warmup_steps),
        lr_decay_steps_(lr_decay_steps) {}

  double step_lr() {
    // Use linear warmup for the initial part
    if (lr_warmup_steps_ > 0 && now_steps_ <= lr_warmup_steps_) {
      now_steps_++;
      return (init_lr_ + (max_lr_ - init_lr_) * now_steps_ / lr_warmup_steps_);
    }

    // For any steps larger than lr_decay_steps, use self.min_lr
    if (now_steps_ > lr_decay_steps_) {
      now_steps_++;
      return min_lr_;
    }

    // If we are done with the warmup period, use the consine decay style
    double num_step = now_steps_ - lr_warmup_steps_;
    double decay_step = lr_decay_steps_ - lr_warmup_steps_;
    double decay_ratio = num_step / decay_step;
    double delta_lr = max_lr_ - min_lr_;

    double coeff = 0.5 * (cos(M_PIl * decay_ratio) + 1.0);

    now_steps_++;

    return min_lr_ + coeff * delta_lr;
  }

 private:
  double init_lr_ = 0.0;
  double max_lr_ = 0.0006;  // 1.5e-4
  double min_lr_ = 1e-5;

  int lr_warmup_steps_ = 700;
  int now_steps_ = 0;  // 0
  int lr_decay_steps_ = 320000;
};

constexpr bool need_infer = false;   // 是否需要训练结束后推理
constexpr int EPOCHS = 1;            // 最大训练轮数
constexpr int num_layers = 48;       // 层数 12
constexpr int embed_dim = 1600;      // embedding大小 768
constexpr int num_heads = 25;        // 注意力头数12
constexpr int seq_length = 1024;     // 最大序列长1024
constexpr int vocab_size = 50257;    // 词典量
constexpr float dropout_rate = 0.0;  // 丢弃率
constexpr int batch_size = 8;       // batch size

// constexpr int total_batch_size = 8;

constexpr int NUM_MICRO_BATCHES = 8;

// 输入数据的形状
auto DATASHAPE = TShape({batch_size / NUM_MICRO_BATCHES, seq_length});
auto LABELSHAPE = TShape({batch_size / NUM_MICRO_BATCHES, seq_length});
auto ATTN_MASK_SHAPE = TShape({seq_length, seq_length});

// FeedForward
Symbol FeedForward(const std::string& name, Symbol X, float dp_rate,
                   int _embed_dim, int nx) {
  auto c_fc = FullyConnected("c_fc" + name, X, nx, false);

  auto act = GELU("act" + name, c_fc, "tanh");  // 出来形状为(32,3072,1024)

  auto c_proj = FullyConnected("c_proj" + name, act, _embed_dim, false);
  auto dropout = Dropout("dropout" + name, c_proj, dp_rate);

  return dropout;
}

// Transformer Decoder Block
Symbol TransformerBlock(Symbol X, const std::string& prefix, Symbol attn_mask,
                        std::map<std::string, Tensor>& argmap, Context& ctx) {
  Symbol sum1, sum2;

  auto ln1 = LayerNormalization("ln1" + prefix, X, 2);

  auto attn = MultiheadAttentionV2(
      "attn" + prefix, ln1,
      {batch_size / NUM_MICRO_BATCHES, seq_length, embed_dim}, ln1,
      {batch_size / NUM_MICRO_BATCHES, seq_length, embed_dim}, ln1,
      {batch_size / NUM_MICRO_BATCHES, seq_length, embed_dim}, embed_dim,
      num_heads, argmap, ctx, attn_mask, ATTN_MASK_SHAPE, std::nullopt,
      std::nullopt, dropout_rate, true, false, false, 0, 0, true, true, true,
      false);

  auto attn_dropped = Dropout("attn_dropout_post" + prefix, attn, dropout_rate);

  sum1 = Sum("ln1_sum" + prefix, 2, {attn_dropped, X});
  // LayerNorm
  auto ln2 = LayerNormalization("ln2" + prefix, sum1, 2);

  // FeedForward
  auto ffw =
      FeedForward("ffw" + prefix, ln2, dropout_rate, embed_dim, embed_dim * 4);
  // Add&Sum
  sum2 = Sum("ffw_sum" + prefix, 2, {sum1, ffw});

  return sum2;
}

// GPT2 Block
Symbol GPT2(std::map<std::string, Tensor>& argmap, Context& ctx) {
  // token序列，形状[batch_size, seq_len]
  auto data = Symbol::Variable("data");
  auto pos = Symbol::Variable("pos");
  auto attention_mask = Symbol::Variable("attention_mask");

  // 词嵌入，输入为数字化的token序列；输出(batch_size,
  // seq_len,embed_dim)，token对应的词向量
  auto wte = EmbeddingV1("wte", data, vocab_size, embed_dim);
  // pos=debugOP("debug1",pos,"1","0");

  // 位置嵌入，输入为token序列中数字的位置，eg：0,1，。。。，1023
  // 输出(batch_size, seq_len,embed_dim)，token对应的位置向量
  auto wpe = EmbeddingV1("wpe", pos, seq_length, embed_dim);
  // wpe=debugOP("debug2",wpe,"0","1");

  // 两个embedding加和
  auto wtpes = Sum("wtpes", 2, {wte, wpe});
  // dropout
  auto blkin = Dropout("blkin", wtpes, dropout_rate);

  // GTP2-XL 48层Transformer Decoder
  auto block = TransformerBlock(blkin, "0", attention_mask, argmap, ctx);
  for (int i = 1; i < num_layers; i++) {
    auto newblock =
        TransformerBlock(block, std::to_string(i), attention_mask, argmap, ctx);
    block = newblock;
  }
  block = LayerNormalization("ln_final", block, 2);
  auto fclogit_weight = Symbol::Variable("fclogit_weight");
  auto t_fclogit_weight = Transpose("t_fclogit_weight", fclogit_weight, {1, 0});
  auto fclogit = MatMul("fclogit", block, t_fclogit_weight);

  auto logits =
      Reshape("logits", fclogit,
              {(seq_length)*batch_size / NUM_MICRO_BATCHES, vocab_size});
  auto label = Symbol::Variable("label0");
  auto labels =
      Reshape("label_s", label, {batch_size / NUM_MICRO_BATCHES * seq_length});
  auto loss = CrossEntropyLoss("loss", logits, labels);
  return loss;
}

exec::Executor* exec_model = nullptr;

cxxopts::ParseResult parse_commandline(int argc, char** argv) {
  cxxopts::Options options(argv[0], "GPT2 训练程序");
  // clang-format off
  options.add_options()
          ("l,lr", "Learning rate", cxxopts::value<float>()->default_value("0.0001"))
          ("g,gpu", "GPU id", cxxopts::value<int>()->default_value("0"))
          ("a,acc", "Accumulation step", cxxopts::value<int>()->default_value("1"))
          ("o,output", "Saved model name", cxxopts::value<std::string>()->default_value(argv[0]));
  // clang-format on
  return options.parse(argc, argv);
}

void create_dir(std::string dataset) {
  try {
    // 删除文件夹及其中的所有文件
    std::filesystem::remove_all(dataset);
    std::cout << dataset << "文件夹及其中的所有文件已删除。" << std::endl;
  } catch (const std::filesystem::filesystem_error& e) {
    std::cout << "删除文件夹失败: " << e.what() << std::endl;
  }
  if (!std::filesystem::exists(dataset)) {
    if (std::filesystem::create_directories(dataset)) {
      std::cout << "create " + dataset << " success!" << std::endl;
    } else {
      std::cout << "create" + dataset + " failed!" << std::endl;
    }
  }
}
// 绑定输入
std::map<std::string, Tensor> args_map, grads_acc;
int main(int argc, char** argv) {
  dmlc::SetEnv("NEXUS_NO_REUSE_MEM", true);
  nexus::distributed::init_process_group("nccl");
  int world_rank = nexus::distributed::get_rank();
  int world_size = nexus::distributed::get_world_size();
  int local_rank = world_rank % nexus::Context::GetGPUCount();

  nexus::distributed::parallel_state::initialize_model_parallel(1, 8);

  std::vector<size_t> ranks;
  ranks.emplace_back(
      distributed::parallel_state::get_pipeline_model_parallel_first_rank());
  ranks.emplace_back(
      distributed::parallel_state::get_pipeline_model_parallel_last_rank());
  auto embedding_group =
      distributed::new_group(ranks, std::chrono::milliseconds(60000));

  auto parsed_options = parse_commandline(argc, argv);
  int accumulation_steps = parsed_options["acc"].as<int>();  // 梯度累加步进

  auto gpu = Context::GPU(local_rank);
  auto cpu = Context::CPU();

  auto nn = GPT2(args_map, gpu);

  args_map["attention_mask"] = Tensor({1024, 1024}, kFloat32, gpu, true);
  args_map["data"] = Tensor(DATASHAPE, kFloat32, gpu, true);
  args_map["fclogit_weight"] = Tensor({vocab_size, embed_dim}, kFloat32, gpu, true);
  args_map["label0"] = Tensor({batch_size / NUM_MICRO_BATCHES, seq_length}, kFloat32, gpu, true);
  args_map["pos"] = Tensor({seq_length}, kFloat32, gpu, true);

  nn.InferArgsMap(gpu, &args_map, args_map, true);

  if (!distributed::parallel_state::is_pipeline_first_stage() && !distributed::parallel_state::is_pipeline_last_stage())
    args_map["attention_mask"].LoadNPY(load_weight_path + "attention_mask.npy");  // 路径
  
  if (distributed::parallel_state::is_pipeline_first_stage())
    args_map["pos"].LoadNPY(load_weight_path + "pos.npy");

  if (distributed::parallel_state::is_pipeline_last_stage())
    args_map["fclogit_weight"] << 0.;

  Engine::Get()->WaitForAll();


  std::vector<std::pair<std::string, std::string>> split_strategy;
  split_strategy.emplace_back(std::make_pair("data", "blkin"));
  split_strategy.emplace_back(std::make_pair("ln10", "ffw_sum7"));
  split_strategy.emplace_back(std::make_pair("ln18", "ffw_sum15"));
  split_strategy.emplace_back(std::make_pair("ln116", "ffw_sum23"));
  split_strategy.emplace_back(std::make_pair("ln124", "ffw_sum31"));
  split_strategy.emplace_back(std::make_pair("ln132", "ffw_sum39"));
  split_strategy.emplace_back(std::make_pair("ln140", "ffw_sum47"));
  split_strategy.emplace_back(std::make_pair("ln_final", "loss"));

  nexus::distributed::_1F1BScheduler _1F1B_exec(
      nn, args_map, gpu, NUM_MICRO_BATCHES, batch_size, split_strategy);

  for (auto& i : args_map) {
    if (i.first == "attention_mask" || i.first == "data" ||
        i.first == "label0" || i.first == "fclogit_weight" ||
        i.first == "pos") {
      continue;
    }
    // 只有当前的stage有这个参数时，才进行load。
    if (_1F1B_exec.HasArgByName(i.first))
      i.second.LoadNPY(load_weight_path + i.first +
                       ".npy");  // P_weightXL/ xlHg/
  }

  // 在外面手动实现权重复用
  if (distributed::parallel_state::is_pipeline_first_stage()) {
    distributed::all_reduce(args_map["wte_embedding"],
                            distributed::ReduceOp::SUM, embedding_group);
  } else if (distributed::parallel_state::is_pipeline_last_stage()) {
    distributed::all_reduce(args_map["fclogit_weight"],
                            distributed::ReduceOp::SUM, embedding_group);
  }

  auto dataloader = std::make_shared<GPT2_Data_Loader>();
  dataloader->Init(batch_size, gpu);

  auto lr_scheduler = Anneals_Cosine_LR_Scheduler();
  double learning_rate = lr_scheduler.step_lr();
  double weight_decay = 1e-1;
  Optimizer* tmp = OptimizerRegistry::Find("adamw");

  tmp->SetParam("lr", learning_rate)->SetParam("wd", weight_decay);
  int count = 0;

  // 时间
  auto now = std::chrono::system_clock::now();
  auto start = std::chrono::steady_clock::now();
  auto t_now = std::chrono::system_clock::to_time_t(now);
  auto tm_now = std::localtime(&t_now);
  std::cout << "当前时间： ";
  std::stringstream ss;
  ss << std::put_time(tm_now, "%Y-%m-%d %H:%M:%S");
  std::string timestr = ss.str();  // 把ss转换成string的对象。
  std::cout << timestr << std::endl;
  double accumulate_loss = 0.0;
  auto saved_model_name = parsed_options["output"].as<std::string>();
  nvtx3::scoped_range r{"main"};
  int real_step = 0;
  for (int iter = 1; iter <= EPOCHS; ++iter) {
    dataloader->BeforeFirst();
    while (dataloader->Next()) {
      std::map<std::string, Tensor> inputs;
      io::TensorBatch batch = dataloader->Value();

      inputs["data"] = batch.data[0];
      inputs["label0"] = batch.data[1];

      auto loss_mp = _1F1B_exec.step(inputs);

      if (distributed::parallel_state::is_pipeline_last_stage()) {
        auto loss = tensor::Concat(loss_mp, 0);
        std::cout << loss << std::endl;
        auto loss_mean = loss.Mean().Item();
        accumulate_loss += loss_mean;
      }

      if ((count + 1) % accumulation_steps == 0) {
        accumulate_loss /= accumulation_steps;

        // 手动对复用的权重进行梯度的allreduce，即训练开始前对两个权重进行同步，且每一次梯度更新前对这两个权重使用的梯度进行allreduce，以此实现权重复用的目的
        if (distributed::parallel_state::is_pipeline_first_stage()) {
          distributed::all_reduce(_1F1B_exec.GetGradByName("wte_embedding"),
                                  distributed::ReduceOp::SUM, embedding_group);
        } else if (distributed::parallel_state::is_pipeline_last_stage()) {
          distributed::all_reduce(_1F1B_exec.GetGradByName("fclogit_weight"),
                                  distributed::ReduceOp::SUM, embedding_group);
        }

        int j = 0;
        for (auto& i : nn.ListArguments()) {
          if (i == "attention_mask" || i == "data" || i == "label0" ||
              i == "fclogit_weight" || i == "pos")
            continue;
          if (_1F1B_exec.HasGradByName(i)) {
            _1F1B_exec.GetGradByName(i) /= NUM_MICRO_BATCHES;
            tmp->Update(j, _1F1B_exec.GetArgByName(i),
                        _1F1B_exec.GetGradByName(i), gpu);
            j++;
          }
        }
        learning_rate = lr_scheduler.step_lr();
        tmp->SetParam("lr", learning_rate);

        _1F1B_exec.zero_grads();

        if (distributed::parallel_state::is_pipeline_last_stage()) {
          std::cout << "iteration: " << count / (1 * accumulation_steps)
                    << " | loss: " << std::fixed << std::setprecision(6)
                    << accumulate_loss;
          accumulate_loss = 0.0f;
          auto end = std::chrono::steady_clock::now();
          auto dt = end - start;
          start = end;
          auto el_time = (double)dt.count() / (1000 * 1000 * 1000);
          std::cout << " | elapsed time(s): " << el_time;
          std::cout << " | throughput(tokens/s): "
                    << 1.0 * batch_size * seq_length * accumulation_steps /
                           el_time
                    << std::endl;
        }
        real_step += 1;
      }

      count++;
      Engine::Get()->WaitForAll();
      return 0;
    }
  }

  // 重要：等待引擎上所有计算完成
  Engine::Get()->WaitForAll();

  Engine::Get()->WaitForAll();
  // 通知引擎关闭
  Engine::Get()->NotifyShutdown();

  return 0;
}