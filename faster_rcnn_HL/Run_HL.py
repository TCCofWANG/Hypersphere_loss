import os

#os.system("python ./train.py --loss_weight 0.001")  #0.758
#print("loss_weight_0.001 finished")

#os.system("python ./train.py --loss_weight 0.003")  #0.762
#print("loss_weight_0.003 finished")

#os.system("python ./train.py --loss_weight 0.004")  #0.758
#print("loss_weight_0.004 finished")

#os.system("python ./train.py --loss_weight 0.005")  #0.760
#print("loss_weight_0.005 finished")


#
# os.system("python ./train.py --loss_weight 0.006")  #
# print("loss_weight_0.006 finished")


os.system("python ./train.py --loss_weight 0.003 --rho 0.07")  #760
print("rho_0.07 finished")

os.system("python ./train.py --loss_weight 0.003 --rho 0.09")  #757
print("rho_0.09  finished")


os.system("python ./train.py --loss_weight 0.003 --rho 0.11")  #
print("rho_0.11 finished")


os.system("python ./train.py --loss_weight 0.003 --rho 0.13")  #
print("rho_0.13  finished")


os.system("python ./train.py --loss_weight 0.003 --rho 0.15")  #
print("rho_0.15 finished")

