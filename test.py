def addToList():
    global a
    currentObject = object()
    currentObject.value1 = 0
    currentObject.value2 = 0
    a.append(currentObject)

class object:
    def __init__(self):
        self.value1 = None
        self.value2 = None

a = []

addToList()
addToList()

print("a:")