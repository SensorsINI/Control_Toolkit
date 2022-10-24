##  Description of Provided Application-Specific Controllers

- `do-mpc`:
    based on do-mpc library, contnuous model, we provide do-mpc library with true equations, it internally integrates it with cvodes
    Example of working parameters: dt=0.2, horizon=10, working parameters from git revision number:

- `do-mpc-discrete`:
    Same as do-mpc, just discrete model obtained from continuous with single step Euler stepping

- `lqr`:
    linear quadratic regulator controller, our very first well working controller
