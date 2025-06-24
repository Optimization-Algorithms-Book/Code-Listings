# 🛠 Errata

This file contains errata for the [Optimization Algorithms: AI techniques for design, planning, and control problems](https://www.manning.com/books/optimization-algorithms) book.

---

### Chapter 1, Section 1.3, Page 8
- The *weak local mimimum* definition is flawed: The mathematical part (*Mathematically,
a point … for all $x$ in $N$*) is correct, but the preceding text and the following image are not. The mention of *sequences of points converging to this point* just makes no sense, and this description doesn’t meet the mathematical definition. Further, Fig. 1.3 shows a saddle point which [**is not** a minimum at all](https://mathworld.wolfram.com/SaddlePoint.html); no matter weak or strong. This could be fixed by turning the saddle point into real weak minima area, like this:

<img src="images/Fig. 1.3.png" />

### Chapter 2, Section 2.1.1, Page 28
- `n!` should be replaced by `(n-1)!`, so *"there are (n-1)! = 12! = 479,001,600 possible tours in the case of ATSP"*.

### Chapter 4, Section 4.3.1, Page 119, Figure 4.17
- Numbers **6, 7, and 8** are missing in the last row of the goal node.

### Chapter 6, Section 6.6, Page 226
- *"As the three swaps"* should be *"As the four swaps`"*.

### Chapter 5, Section 5.2.2, Page 164, Algorithm 5.1
- *"new location/state `x_n+1`"* should be *"new location/state `x_{n+1}`"*.

### Chapter 7, Section 7.3.3, Page 256, Figure 7.12
- The final selected individual is **x = 110, f(x) = 90000**.

### Appendix A, Section A.2.1, Page 497
- *"car manufacturing problem"* → *"chess set problem"*.

---
