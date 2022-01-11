import torch
d = int(3)

#quasistatic version of the efem, assuming no mass:

class ElasticFEM:
  def __init__(self, mesh, X):
    self.mesh = mesh
    self.Initialize(X)
    
  def Initialize(self, X):
    self.Np = int(X.size(0)/3)
    elements = int(len(self.mesh)/(d+1))
    self.measure = torch.zeros(elements)
    self.Dm_inverse = torch.zeros(d*d*elements)
    one_over_d_factorial = 1.0
    for i in range(d):
      one_over_d_factorial = one_over_d_factorial/(i+1)
    for e in range(elements):
      Dm = self.Ds(e,X)
      self.measure[e] = one_over_d_factorial * Dm.det()
      assert self.measure[e] > 0, 'degenerate element detected'
      Dm_inv = torch.linalg.inv(Dm)
      for r in range(d):
        for c in range(d):
          self.Dm_inverse[(d*d)*e+d*r+c] = Dm_inv[r, c]

  def F(self, e, x):
    F = torch.matmul(self.Ds(e,x), self.ElementDmInv(e))
    return F

  def ElementDmInv(self, e):
    Dm_inv = torch.zeros(d,d)
    for r in range(d):
      for c in range(d):
        Dm_inv[r,c] = self.Dm_inverse[(d * d) * e + d * r + c]
    return Dm_inv

  def Ds(self, e, u):
    result = torch.zeros(d,d)
    for i in range(d):
      for c in range(d):
        result[c,i] = u[d*self.mesh[(d + 1) * e + i + 1] + c] - u[d*self.mesh[(d + 1) * e]+ c]
    return result

  def potential_energy(self, psi, x, mu, lam):
    total = torch.zeros(1)
    for e in range(len(self.measure)):
      total = total + self.measure[e] * psi(self.F(e, x), mu, lam)
    return total

  def add_internal_force(self, x, P, incident_elements, mu, lam, scale = 1):
    residual = torch.zeros(len(x))
    if len(incident_elements) == 0:
      elements = len(self.measure)
      for e in range(elements):
        Fe = self.F(e, x)
        Pe = P(Fe, mu, lam)
        g = -self.measure[e]*torch.matmul(Pe, self.ElementDmInv(e).transpose(0,1))
        for ie in range(d):
          inverse_mass = 1
          for c in range(d):
            residual[d*self.mesh[(d + 1) * e + ie + 1]+c] = residual[d*self.mesh[(d + 1) * e + ie + 1]+c] + scale*inverse_mass*g[c, ie]

    return residual



