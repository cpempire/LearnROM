import numpy as np
import dolfin as dl

def SamplesProjection(nModes=0, test_samples=None, nSamples=1, sampling=None):

    samples = []

    for n in range(nSamples):
        sample = 0.*sampling.U[:,0]

        if sampling.distribution is 'uniform':
            for i in range(nModes):
                sample += np.dot(test_samples[n], np.dot(sampling.HiFi.CovInv, sampling.U[:,i]))*sampling.U[:,i]
            samples.append(sample)
        elif sampling.distribution is 'gaussian':
            sample_n = sampling.pde.generate_parameter()
            sample_n.set_local(test_samples) # only one sample a time
            for i in range(nModes):
                # sample += self.RandomSamples[n,i]*self.U[:,i]*np.sqrt(self.rho[i])
                U_i = sampling.pde.generate_parameter()
                U_i.set_local(sampling.U[:,i])
                sampling.HiFi.Gauss.R.inner(sample_n, U_i)
                sample += sampling.HiFi.Gauss.R.inner(sample_n, U_i)*sampling.U[:,i]
            samples.append(sample)

    samples = np.array(samples)

    return samples

def ErrorParameterReduction(HiFi, samples, nSamples=1, sampling=None, nPRset=1, qoi=None, plot=False):

    error_pr = np.zeros([nSamples,sampling.nTotalModes+1])
    if qoi is not None:
        error_pr_q = np.zeros([nSamples,sampling.nTotalModes+1])

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

            for n in nPRset:
                samples_pr = SamplesProjection(nModes=n, test_samples=samples, nSamples=nSamples, sampling=sampling)
                u_pr = HiFi.snapshots(samples_pr)
                if qoi is not None:
                    q_pr = np.zeros(nSamples)
                    for i in range(nSamples):
                        u = dl.Function(HiFi.Vh[0]).vector()
                        u.set_local(u_pr[:,i])
                        m = dl.Function(HiFi.Vh[1]).vector()
                        m.set_local(samples_pr[i])
                        x = (u, m)
                        q_pr[i] = qoi.eval(x)

                u_diff = u_hifi - u_pr
                for i in range(nSamples):
                    # print "HiFi.Xnorm.array()", HiFi.Xnorm.array().shape, u_diff[:,i].shape
                    error_pr[i,n] = np.sqrt(np.dot(u_diff[:,i].T, HiFi.Xnorm.dot(u_diff[:,i])))\
                                /np.sqrt(np.dot(u_hifi[:,i].T, HiFi.Xnorm.dot(u_hifi[:,i])))
                    if qoi is not None:
                        error_pr_q[i,n] = np.abs(q_hifi[i] - q_pr[i])/np.abs(q_hifi[i])

        elif HiFi.problem_type is 'nonaffine':
            done = True
            for i in range(nSamples):
                sample = samples[i]
                u_hifi = HiFi.solve(sample) # hifi matrix and vector is also assembled for RB projection
                for n in nPRset:

                    # sample_pr = SamplesProjection(nModes=n, test_samples=sample, nSamples=1, sampling=sampling)
                    sample_pr = sampling.SubspaceProjection(nModes=n, sample=sample)
                    u_pr = HiFi.solve(sample_pr) # do not need to assemble again
                    u_diff = u_hifi-u_pr
                    error_pr[i,n] = np.sqrt(np.dot(u_diff, HiFi.Xnorm.dot(u_diff)))\
                        /np.sqrt(np.dot(u_hifi, HiFi.Xnorm.dot(u_hifi)))
                    if qoi is not None:
                        u = dl.Function(HiFi.Vh[0]).vector()
                        u.set_local(u_hifi)
                        m = dl.Function(HiFi.Vh[1]).vector()
                        m.set_local(sample)
                        x = (u,m)
                        q_hifi = qoi.eval(x)
                        u = dl.Function(HiFi.Vh[0]).vector()
                        u.set_local(u_pr)
                        m = dl.Function(HiFi.Vh[1]).vector()
                        m.set_local(sample_pr)
                        x = (u,m)
                        q_pr = qoi.eval(x)
                        error_pr_q[i,n] = np.abs(q_hifi - q_pr)/np.abs(q_hifi)

    if done is True:
        error_pr_max = np.max(error_pr, axis=0)
        error_pr_mean = np.mean(error_pr,axis=0)

        if qoi is not None:
            error_pr_max_q = np.max(error_pr_q,axis=0)
            error_pr_mean_q= np.mean(error_pr_q,axis=0)


    if qoi is not None:
        return error_pr_max, error_pr_mean, error_pr_max_q, error_pr_mean_q
    else:
        return error_pr_max, error_pr_mean