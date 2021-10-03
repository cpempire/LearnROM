import numpy as np
import dolfin as dl

def errorRB(HiFi, RB, samples, nRBset=None, qoi=None, plot=False):

    if plot:
        import matplotlib.pyplot as plt

    nSamples = samples.shape[0]

    if nRBset is None:
        nRBset = range(1,RB.N)
    
    error_rb = np.zeros([nSamples,RB.N])
    if qoi is not None:
        error_rb_q = np.zeros([nSamples,RB.N])

    # for nonaffine problems
    done = False
    if hasattr(HiFi,'problem_type'):
        # for affine problems
        if HiFi.problem_type is 'affine':
            done = True
            u_hifi = HiFi.snapshots(samples)
            if qoi is not None:
                q_hifi = np.zeros(nSamples)
                for i in range(nSamples):
                    u = dl.Function(HiFi.Vh[0]).vector()
                    u.set_local(u_hifi[:,i])
                    m = dl.Function(HiFi.Vh[1]).vector()
                    m.set_local(samples[i,:])
                    x = (u, m)
                    q_hifi[i] = qoi.eval(x)

            for n in nRBset:
                u_rb = RB.snapshots(samples, n) # reconstruct a hifi solution as V^T*uN
                if qoi is not None:
                    q_rb = np.zeros(nSamples)
                    for i in range(nSamples):
                        u = dl.Function(HiFi.Vh[0]).vector()
                        u.set_local(u_rb[:,i])
                        m = dl.Function(HiFi.Vh[1]).vector()
                        m.set_local(samples[i,:])
                        x = (u, m)
                        q_rb[i] = qoi.eval(x)

                u_diff = u_hifi - u_rb
                for i in range(nSamples):
                    error_rb[i,n] = np.sqrt(np.dot(u_diff[:,i].T, HiFi.Xnorm.dot(u_diff[:,i])))\
                                /np.sqrt(np.dot(u_hifi[:,i].T, HiFi.Xnorm.dot(u_hifi[:,i])))
                    if qoi is not None:
                        error_rb_q[i,n] = np.abs(q_hifi[i] - q_rb[i])/np.abs(q_hifi[i])

        elif HiFi.problem_type is 'nonaffine':
            done = True
            for i in range(nSamples):
                sample = samples[i,:]
                u_hifi = RB.HiFi.solve(sample) # hifi matrix and vector is also assembled for RB projection
                for n in nRBset:
                    if n == nRBset[0]:
                        assemble = True
                    else:
                        assemble = False
                    u_rb = RB.reconstruct(sample, n, assemble=assemble) # do not need to assemble again
                    u_diff = u_hifi-u_rb
                    error_rb[i,n] = np.sqrt(np.dot(u_diff, HiFi.Xnorm.dot(u_diff)))\
                        /np.sqrt(np.dot(u_hifi, HiFi.Xnorm.dot(u_hifi)))
                    if qoi is not None:
                        u = dl.Function(HiFi.Vh[0]).vector()
                        u.set_local(u_hifi)
                        m = dl.Function(HiFi.Vh[1]).vector()
                        m.set_local(sample)
                        x = (u,m)
                        q_hifi = qoi.eval(x)
                        u = dl.Function(HiFi.Vh[0]).vector()
                        u.set_local(u_rb)
                        m = dl.Function(HiFi.Vh[1]).vector()
                        m.set_local(sample)
                        x = (u,m)
                        q_rb = qoi.eval(x)
                        error_rb_q[i,n] = np.abs(q_hifi - q_rb)/np.abs(q_hifi)

    if done is True:
        error_rb_max = np.max(error_rb, axis=0)
        error_rb_mean = np.mean(error_rb,axis=0)

        if qoi is not None:
            error_rb_max_q = np.max(error_rb_q,axis=0)
            error_rb_mean_q= np.mean(error_rb_q,axis=0)

        if plot:
            plt.figure()
            plt.semilogy(nRBset, error_rb_max[nRBset],'*-', label='error_rb_max_u')
            plt.semilogy(nRBset, error_rb_mean[nRBset],'rx-',label='error_rb_mean_u')
            if qoi is not None:
                plt.semilogy(nRBset, error_rb_max_q[nRBset],'gd-', label='error_rb_max_q')
                plt.semilogy(nRBset, error_rb_mean_q[nRBset],'ks-',label='error_rb_mean_q')
            plt.legend()
            plt.show()


    # for nonlinear problems
    if done is not True:
        u_hifi = HiFi.snapshots(samples)
        error_rb_eim = np.zeros([nSamples,RB.N])

        if qoi is not None:
            q_hifi = np.zeros(nSamples)
            for i in range(nSamples):
                u = dl.Function(HiFi.Vh[0]).vector()
                u.set_local(u_hifi[:,i])
                m = dl.Function(HiFi.Vh[1]).vector()
                m.set_local(samples[i,:])
                x = (u, m)
                q_hifi[i] = qoi.eval(x)

            error_rb_eim_q = np.zeros([nSamples,RB.N])

        for n in nRBset:
            u_rb = RB.snapshots(samples, n, output='solution') # reconstruct a hifi solution as V^T*uN
            if qoi is not None:
                q_rb = np.zeros(nSamples)
                for i in range(nSamples):
                    u = dl.Function(HiFi.Vh[0]).vector()
                    u.set_local(u_rb[:,i])
                    m = dl.Function(HiFi.Vh[1]).vector()
                    m.set_local(samples[i,:])
                    x = (u, m)
                    q_rb[i] = qoi.eval(x)

            u_diff = u_hifi - u_rb
            for i in range(nSamples):
                error_rb[i,n] = np.sqrt(np.dot(u_diff[:,i].T, HiFi.Xnorm.dot(u_diff[:,i])))\
                            /np.sqrt(np.dot(u_hifi[:,i].T, HiFi.Xnorm.dot(u_hifi[:,i])))
                if qoi is not None:
                    error_rb_q[i,n] = np.abs(q_hifi[i] - q_rb[i])/np.abs(q_hifi[i])

            # with hyper reduction
            # print "range", range(RB.solver.DEIM.N-1, RB.solver.DEIM.N)

            # for m in range(RB.solver.DEIM['N']-1, RB.solver.DEIM['N']):
            m = RB.solver.DEIM.N
            u_rb_eim = RB.snapshotsDEIM(samples, n, m, output='solution')
            if qoi is not None:
                q_rb_eim = np.zeros(nSamples)
                for i in range(nSamples):
                    u = dl.Function(HiFi.Vh[0]).vector()
                    u.set_local(u_rb_eim[:,i])
                    m = dl.Function(HiFi.Vh[1]).vector()
                    m.set_local(samples[i,:])
                    x = (u, m)
                    q_rb_eim[i] = qoi.eval(x)

            u_diff = u_hifi - u_rb_eim
            for i in range(nSamples):
                # print "HiFi.Xnorm.array()", HiFi.Xnorm.array().shape, u_diff[:,i].shape
                error_rb_eim[i,n] = np.sqrt(np.dot(u_diff[:,i].T, HiFi.Xnorm.dot(u_diff[:,i])))\
                            /np.sqrt(np.dot(u_hifi[:,i].T, HiFi.Xnorm.dot(u_hifi[:,i])))
                if qoi is not None:
                    error_rb_eim_q[i,n] = np.abs(q_hifi[i] - q_rb_eim[i])/np.abs(q_hifi[i])


        error_rb_max = np.max(error_rb, axis=0)
        error_rb_mean = np.mean(error_rb,axis=0)
        error_rb_eim_max = np.max(error_rb_eim, axis=0)
        error_rb_eim_mean = np.mean(error_rb_eim,axis=0)

        if qoi is not None:
            error_rb_max_q = np.max(error_rb_q,axis=0)
            error_rb_mean_q= np.mean(error_rb_q,axis=0)
            error_rb_eim_max_q = np.max(error_rb_eim_q,axis=0)
            error_rb_eim_mean_q= np.mean(error_rb_eim_q,axis=0)

        if plot:
            plt.figure()
            plt.semilogy(nRBset, error_rb_max[nRBset],'b*-', label='error_rb_max_u')
            plt.semilogy(nRBset, error_rb_mean[nRBset],'rx-',label='error_rb_mean_u')
            plt.semilogy(nRBset, error_rb_eim_max[nRBset],'gd-', label='error_rb_eim_max_u')
            plt.semilogy(nRBset, error_rb_eim_mean[nRBset],'ko-',label='error_rb_eim_mean_u')
            plt.legend()

            if qoi is not None:
                plt.figure()
                plt.semilogy(nRBset, error_rb_max_q[nRBset],'b*-', label='error_rb_max_q')
                plt.semilogy(nRBset, error_rb_mean_q[nRBset],'rx-',label='error_rb_mean_q')
                plt.semilogy(nRBset, error_rb_eim_max_q[nRBset],'gd-', label='error_rb_eim_max_q')
                plt.semilogy(nRBset, error_rb_eim_mean_q[nRBset],'ko-',label='error_rb_eim_mean_q')
                plt.legend()

            plt.show()


    if qoi is not None:
        return error_rb_max, error_rb_mean, error_rb_max_q, error_rb_mean_q
    else:
        return error_rb_max, error_rb_mean