import matplotlib.pyplot as plt
WT_n = 0
TC_n = 0
ET_n = 0
WT_x = []
WT_y = []
TC_x = []
TC_y = []
ET_x = []
ET_y = []
with open("/home/server/github/pytorch-3dunet/checkpoints/templatemixup_NoNewNet_BN_Adam_batchsize=2_lr=0.0001/model.log") as f:   #根据自己的目录做对应的修改
    for line in f:
        line = line.strip()
        if len(line.split("Val_WT:")) == 2:
            WT_y.append(float(line.split("Val_WT:")[1]))
            WT_x.append(WT_n)
            WT_n += 300   #根据test_interval调整
        if len(line.split("Val_TC:")) == 2:
            TC_y.append(float(line.split("Val_TC:")[1]))
            TC_x.append(TC_n)
            TC_n += 300   #根据display调整
        if len(line.split("Val_ET:")) == 2:
            ET_y.append(float(line.split("Val_ET:")[1]))
            ET_x.append(ET_n)
            ET_n += 300   #根据display调整
plt.figure(figsize=(8,6))
plt.plot(WT_x,WT_y,'',label="Val_WT:")
plt.plot(TC_x,TC_y,'',label="Val_TC:")
plt.plot(ET_x,ET_y,'',label="Val_ET:")
plt.title('val_dice')
plt.legend(loc='upper right')
plt.xlabel('iter')
plt.ylabel('dice')
plt.grid(WT_x)
plt.show()
