This project was heavily inspired by [a blog post by John Baez](https://johncarlosbaez.wordpress.com/2011/12/11/the-beauty-of-roots). In the blog post, John Baez discusses the *Littlewood polynomials*, the set of monic polynomials whose coefficients are either $1$ or $-1$. In particular, he generated the roots of all the Littlewood polynomials of degree 24, which took him 4 days (albeit on 2006 hardware).

I stumbled upon the diagram that he made, and decided to try and optimise the creation process. As finding a single polynomial's roots is computationally inexpensive, one can easily have a GPU calculate the roots of thousands of polynomials, at the same time, in milliseconds. As a result, I wrote a program using Numba and pyCUDA to heavily parallelise the root-finding workload, reducing the time to find the roots down to around 40 seconds.

I also decided to extend the idea of Littlewood polynomials to *Generalised Littlewood Polynomials*, monic polynomials whose coefficients are in the set $\{a, 1\}$ for some parameter $a$. This program is able to generate similar diagrams for Generalised Littlewood polynomials, by varying the parameter `a`. By doing so, I hope to better understand the properties of the roots of the Generalised Littlewood Polynomials.


