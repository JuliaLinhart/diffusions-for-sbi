import torch

from sbi.utils import BoxUniform

parameters = [
    ("C", 10.0, 250.0),
    ("mu", 50.0, 500.0),
    ("sigma", 100.0, 5000.0),
    ("gain", -20.0, +20.0),
]


class prior_JRNMM(BoxUniform):
    def __init__(self, parameters=parameters):
        self.parameters = parameters

        self.bounds = {param[0]: [param[1], param[2]] for param in parameters}

        self.low = torch.tensor([v[0] for v in self.bounds.values()])
        self.high = torch.tensor([v[1] for v in self.bounds.values()])

        super().__init__(low=self.low, high=self.high)

    def condition(self, gain):
        """
        This functions returns the prior distribution for [C, mu, sigma]
        parameter. It is written like this for compatibility purposes with
        the Pyro framework
        """

        low = []
        high = []
        for i in range(len(self.parameters)):
            if self.parameters[i][0] == "gain":
                pass
            else:
                low.append(self.parameters[i][1])
                high.append(self.parameters[i][2])
        low = torch.tensor(low, dtype=torch.float32)
        high = torch.tensor(high, dtype=torch.float32)
        return BoxUniform(low=low, high=high)


if __name__ == "__main__":
    prior = prior_JRNMM()

    theta = prior.sample((4096,))

    print(prior.bounds)
    print(prior.low)
    print(prior.high)
    
    low = (prior.low - theta.mean(0)) / theta.std(0) * 2
    high = (prior.high - theta.mean(0)) / theta.std(0) * 2

    print(low)
    print(high)
