
import sys
import os.path
import os

# This is a tiny script to help you creating a CSV file from a face
# database with a similar hierarchie:
#
#  philipp@mango:~/facerec/data/at$ tree
#  .
#  |-- README
#  |-- s1
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  |-- s2
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  ...
#  |-- s40
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#

if __name__ == "__main__":

    #if len(sys.argv) != 2:
     #   print "usage: create_csv <base_path>"
      #  sys.exit(1)

    BASE_PATH="D:\documents\Polytech_2014-2015\Semestre_8\Regie-video-autonome-\Att_faces"#sys.argv[1]#"D:\documents\Polytech_2014-2015\Semestre_8\Projet\Att_faces"
    SEPARATOR=";"
    output = open("at.csv", 'w')
    label = 0
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                abs_path = "%s\%s" % (subject_path, filename)
                print "%s%s%d" % (abs_path, SEPARATOR, label)
                output.write("%s%s%d\n" % (abs_path, SEPARATOR, label))
            label = label + 1
os.system("pause")
