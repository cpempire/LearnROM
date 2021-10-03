# LearnROM

Learn Reduced Order Models in high-dimensional spaces.

Model reduction is known very challenging for high-dimensional parametric problems whose solutions also live in high-dimensional manifolds. However, often the manifold of some quantity of interest (QoI) depending on the parametric solutions is low-dimensional. LearnROM implements structure-exploiting algorithms to efficiently learn the intrinsic parameter subspace in which the QoI is most sensitive. Both the gradient-based active subspace and Hessian-based subspace are implemented in LearnROM. Samples are drawn from such subspaces to learn the QoI-oriented ROM, which are demonstrated to be more efficient than the samples drawn randomly. See more details in the paper 

```
@article{chen2019hessian,
  title={Hessian-based sampling for high-dimensional model reduction},
  author={Chen, Peng and Ghattas, Omar},
  journal={International Journal for Uncertainty Quantification},
  volume={9},
  number={2},
  year={2019},
  publisher={Begel House Inc.}
}
```

<p>
<img src="images/hessian.png" width=30%>
  <img src="images/block256x256_1.png" width=30%>
  <img src="images/porousmedium1.png" width=30%>
</p>
