def newtonMethod(firstDeriv,secondDeriv,starting_x,maxIters,minStep):
    iters=0
    cur_x=starting_x
    step=100
    while iters<maxIters and abs(step)>minStep:
        prev_x=cur_x
        h = firstDeriv(cur_x) /secondDeriv(cur_x)
        cur_x= cur_x - h
        iters = iters + 1  # iteration count
        step= prev_x-cur_x

    print("The local minimum occurs at", cur_x,'after',iters,"iterations")
    return cur_x,iters


df=lambda x: 4*pow((x-5),3)+3
ddf=lambda x:12*pow((x-5),2)

newtonMethod(df,ddf,10,100,1e-9)