from CStaff import CStaff
import sys

if __name__ == '__main__':
    staff = CStaff(['hello', "train.param"])
    # staff = CStaff(sys.argv)
    staff.start()
