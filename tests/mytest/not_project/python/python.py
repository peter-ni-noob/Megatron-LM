
class obj0:
    def __init__(self,val) -> None:
        self.val=val
    def __eq__(self, value: object) -> bool:
        return self.val==value.val

a=obj0(1)
b=obj0(1) 

# a=5
# b=5


print(a==b)
print(a is b)
print(id(a))
print(id(b))