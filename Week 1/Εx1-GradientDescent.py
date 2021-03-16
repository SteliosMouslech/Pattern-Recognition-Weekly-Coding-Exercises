def gradientDescent(df,starting_x,learningRate,maxIters,minStep):
    iters=0
    cur_x=starting_x
    step=100 #dummy value for first iteration
    while  iters < maxIters and abs(step)>minStep:
        prev_x = cur_x  # Store current x value in prev_x
        cur_x = prev_x - learningRate * df(prev_x)  # Grad descent
        iters = iters + 1  # iteration count
        step= prev_x-cur_x

    print("The local minimum is at", cur_x,'after',iters,'iterations')
    return cur_x, iters


df=lambda x: 4*pow((x-5),3)+3
gradientDescent(df,10,0.02,1000,1e-9)