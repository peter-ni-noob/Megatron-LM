import argparse

class obj0:
    pass

    # def __init__(self) -> None:
    #     pass

    # def __str__(self) -> str:
    #     pass    

def get_args():
    parser = argparse.ArgumentParser('nihao')
    parser.add_argument('--batch_size', action='store_true', default=False)
    parser.add_argument('--update_freq', default=1, type=int)
    o=obj0()
    #打印help更好看
    group = parser.add_argument_group(title='network size')
    group.add_argument('--input', help='input file')
    group.add_argument('--output', help='output file')
    return parser.parse_args()




if __name__ == "__main__":
    # a=get_args()
    # d={"dd":5,"de":"8",55:6}
    # print(d.get(55))
    # print(None is None)
    a=obj0()
    print(vars(a))
    # try:
    #     print(a.batch_size)
    # finally:
    #     print("ok")    
