import numpy as np, sys
d=sys.argv[1]
M=np.load(d+"/track.npy")[:,0]; Q=np.load(d+"/query.npy")[:,0]
q=Q[:,[0,1]]
for k,name in enumerate(["enc","encc","p","pc"]):
    yx=M[:,k,:,:2]; mc=M[:,k,:,2]; md=M[:,k,:,3]
    step=np.abs(np.diff(yx,axis=1)).max(-1)
    d0=np.abs(yx[:,0]-q).max(-1)
    print(name,"d(q,slot0) mean %.2f med %.1f  frac<=1 %.2f"%(d0.mean(),np.median(d0),(d0<=1).mean()),
          " step mean %.2f frac<=1 %.2f frac>=4 %.2f"%(step.mean(),(step<=1).mean(),(step>=4).mean()),
          " maxcos s0 %.3f s7 %.3f  med %.3f"%(mc[:,0].mean(),mc[:,7].mean(),md[:,0].mean()))
print("query energy", np.percentile(Q[:,2],[10,50,90]))
