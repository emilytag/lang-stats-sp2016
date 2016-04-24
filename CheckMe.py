'''
Created on Apr 24, 2016

@author: elliotschumacher
'''
import sys
def check():
    with open(sys.argv[1], "r") as predfile, open(sys.argv[2], "r") as actualfile:
        predlines = predfile.readlines()
        actlines = actualfile.readlines()
        correct = 0.0
        for i in range(0, len(actlines)):
            if int(predlines[i].split()[2].strip()) == int(actlines[i].strip()):
                correct += 1.0
        print(correct / len(actlines))
    pass


if __name__ == '__main__':
    check()