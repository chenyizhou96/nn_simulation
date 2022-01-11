import torch
import python_bindings.TGSL as TGSL

d=3

#note: this one is broken because qr in torch is potentially different from the one in our codebase
#way to fix: define qr ourself or figure out gradient of their version
def corotated_psi(F, mu, lam):
  #F.register_hook(print)
  R, S= torch.linalg.qr(F, mode = 'complete')
  b = torch.matmul(F.transpose(0, 1), F)
  J = torch.linalg.det(F)
  psi = mu*mu_term.apply(F) + lam*(J-1)*(J-1)/2.0

  return psi 

def linear_elasticity_psi(F, mu, lam):
  #F.register_hook(print)
  eps = 0.5*(F+F.transpose(0, 1)) - torch.eye(3)
  psi = mu*torch.matmul(eps,eps).trace() + 0.5*lam*eps.trace()*eps.trace()

  return psi

def linear_elasticity_p(F, mu, lam):
  P = mu*(F+F.transpose(0,1) - 2*torch.eye(3)) + lam* (F.trace()-d)*torch.eye(3)
  return P


class mu_term(torch.autograd.Function):
  @staticmethod
  def forward(cls, F):
    F_flat = F.reshape(-1)
    R = []
    S = []
    TGSL.QRDecomp(F_flat, R, S)
    R = torch.FloatTensor(R)
    S = torch.FloatTensor(S)
    R = R.reshape(3,3)
    S = S.reshape(3,3)
    b = torch.matmul(F.transpose(0, 1), F)
    cls.save_for_backward(F, R, S)
    return b.trace()-2*S.trace() + d

  @staticmethod
  def backward(cls, grad):
    F, R, S = cls.saved_tensors
    return 2*(F-R)*grad

