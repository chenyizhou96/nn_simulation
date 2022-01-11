import models

class SimExample:
  def __init__(self,config):
    self.psi = models.linear_elasticity_psi
    if config.model_number == 0:
      self.psi = models.linear_elasticity_psi
    elif config.model_number == 1:
      self.psi = models.corotated_psi
    self.p=models.linear_elasticity_p
    
