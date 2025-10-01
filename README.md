# HEOpt
Heat exchanger design optimization in openMDAO.

## Getting Started

[Crossflow Shell and Tube Heat Exchanger Optimization in openMDAO.pdf](https://github.com/Finn-Arcadi/HEOpt/blob/main/Crossflow%20Shell%20and%20Tube%20Heat%20Exchanger%20Optimization%20in%20openMDAO.pdf) shows example outputs of the optimization, covers all the theory behind the model, and includes some validation against academic sources.

simple_opt.py can be executed to run one of the example optimization problems once prerequisites are met.

### Prerequisites

This project is dependent on the following:

- openMDAO for optimization (via https://openmdao.org/)
- CoolProp for fluid properties (via https://pypi.org/project/coolprop/)
- ht for some helpful libraries for heat transfer calculations (via https://pypi.org/project/ht/).
