def __ComptR1_EA__(nocc,ntot,\
                   x1,x2,t1,t2,Fock,v2ijab,ChVov,ChVvv):
  '''
  Compute R1 contribution, i.e. x1^T ( Hss x1 + Hsd x2 )
  '''
  import numpy as np
  import numpy.linalg as npla

  Ft1v2ijab = Fock[0:nocc,nocc:ntot]+np.einsum('ai,ijab->jb',t1,v2ijab)
  '''
  C i0 ( p2 )_xf + = 1 * Sum ( p6 ) * x ( p6 )_x * i1 ( p2 p6 )_f
  C   i1 ( p2 p6 )_f + = 1 * f ( p2 p6 )_f
  C   i1 ( p2 p6 )_vt + = 1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 p2 p3 p6 )_v
  '''
  interm0  =  np.einsum('am,ba->bm',x1,Fock[nocc:ntot,nocc:ntot])
  #interm1  =  np.einsum('ai,ibac->bc',t1,v2iabc)
  #interm0 +=  np.einsum('cm,bc->bm',x1,interm1)
  tmp1 =  np.einsum('ai,ian->n',t1,ChVov)
  tmp2 =  np.einsum('cm,bcn->bmn',x1,ChVvv)
  interm0 +=  np.einsum('n,bmn->bm',tmp1,tmp2)
  tmp1 =  np.einsum('ai,ban->bin',t1,ChVvv)
  tmp2 =  np.einsum('cm,icn->imn',x1,ChVov)
  interm0 += -np.einsum('bin,imn->bm',tmp1,tmp2)

  '''
  C i0 ( p2 )_xf + = 1 * Sum ( p7 h6 ) * x ( p2 p7 h6 )_x * i1 ( h6 p7 )_f
  C   i1 ( h6 p7 )_f + = 1 * f ( h6 p7 )_f
  C   i1 ( h6 p7 )_vt + = 1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 h6 p3 p7 )_v
  '''
  interm0 +=  np.einsum('abim,ib->am',x2,Ft1v2ijab)

  '''
  C i0 ( p2 )_xv + = -1/2 * Sum ( p4 p5 h3 ) * x ( p4 p5 h3 )_x * v ( h3 p2 p4 p5 )_v
  '''
  #interm0 += -0.5*np.einsum('abim,icab->cm',x2,v2iabc)
  tmp1 =  np.einsum('abim,ian->bmn',x2,ChVov)
  tmp2 =  np.einsum('abim,ibn->amn',x2,ChVov)
  interm0 += -0.5*np.einsum('bmn,bcn->cm',tmp1-tmp2,ChVvv)

  '''
  C i0 ( p2 )_fxt + = -1 * Sum ( h3 ) * t ( p2 h3 )_t * i1 ( h3 )_fx
  C   i1 ( h3 )_fx + = 1 * Sum ( p7 ) * x ( p7 )_x * i2 ( h3 p7 )_f
  C     i2 ( h3 p7 )_f + = 1 * f ( h3 p7 )_f
  C     i2 ( h3 p7 )_vt + = 1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 h3 p4 p7 )_v
  C   i1 ( h3 )_vx + = 1/2 * Sum ( h4 p5 p6 ) * x ( p5 p6 h4 )_x * v ( h3 h4 p5 p6 )_v
  '''
  interm1  =  np.einsum('am,ia->im',x1,Ft1v2ijab)
  interm1 +=  0.5*np.einsum('abim,jiab->jm',x2,v2ijab)
  interm0 += -np.einsum('ai,im->am',t1,interm1)

  '''
  C i0 ( p2 )_vxt + = 1/2 * Sum ( h4 h5 p3 ) * t ( p2 p3 h4 h5 )_t * i1 ( h4 h5 p3 )_vx
  C   i1 ( h4 h5 p3 )_vx + = 1 * Sum ( p6 ) * x ( p6 )_x * v ( h4 h5 p3 p6 )_v
  '''
  interm1  =  np.einsum('am,ijba->ijbm',x1,v2ijab)
  interm0 +=  0.5*np.einsum('abij,ijbm->am',t2,interm1)

  return interm0

def __ComptR2_EA__(nocc,ntot,\
                   x1,x2,t1,t2,Fock,v2ijab,v2ijka,v2iajb,\
                   ChVov,ChVvv):
  '''
  Compute R2 contribution, i.e. x2^T ( Hds x1 + Hdd x2 )
  Note: x (dummy, hx), x ( px dummy hx hx ), and interm0 ( px dummy hx hx )
  '''
  import numpy as np

  Ft1v2ijab =  Fock[0:nocc,nocc:ntot]+np.einsum('ai,ijab->jb',t1,v2ijab)
  t1ChVvv   =  np.einsum('ai,abn->ibn',t1,ChVvv)
  x1ChVvv   =  np.einsum('am,acn->cmn',x1,ChVvv)
  x1v2ijab  =  np.einsum('bm,ijab->ijam',x1,v2ijab)
  x2v2ijab  =  np.einsum('cbjm,ijab->icam',x2,v2ijab)
  x2v2ijab2 =  np.einsum('abkm,ijab->ijkm',x2,v2ijab)
  intermy   = -0.5*np.einsum('am,ijka->ijkm',x1,v2ijka)\
              +0.25*x2v2ijab2\
              -0.5*np.einsum('ak,ijam->ijkm',t1,x1v2ijab)
  '''
  C i0 ( p3 p4 h2 )_xv + = -1 * Sum ( p5 ) * x ( p5 )_x * v ( p3 p4 h2 p5 )_v
  '''
  #interm0  = -np.einsum('am,iabc->bcim',x1,v2iabc)
  tmp  = -np.einsum('ibn,cmn->bcim',ChVov,x1ChVvv)
  interm0  = tmp-np.einsum('cbim->bcim',tmp)

  '''
  C i0 ( p3 p4 h2 )_xf + = -1 * Sum ( h8 ) * x ( p3 p4 h8 )_x * i1 ( h8 h2 )_f
  C   i1 ( h8 h1 )_f + = 1 * f ( h8 h1 )_f
  C   i1 ( h8 h1 )_ft + = 1 * Sum ( p9 ) * t ( p9 h1 )_t * i2 ( h8 p9 )_f
  C     i2 ( h8 p9 )_f + = 1 * f ( h8 p9 )_f
  C     i2 ( h8 p9 )_vt + = 1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 h8 p6 p9 )_v
  C   i1 ( h8 h1 )_vt + = -1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 h8 h1 p5 )_v
  C   i1 ( h8 h1 )_vt + = -1/2 * Sum ( h7 p5 p6 ) * t ( p5 p6 h1 h7 )_t * v ( h7 h8 p5 p6 )_v
  '''
  interm1  =  Fock[0:nocc,0:nocc] + np.einsum('ai,ja->ji',t1,Ft1v2ijab)
  interm1 += -np.einsum('ai,ijka->jk',t1,v2ijka)
  interm1 += -0.5*np.einsum('abij,jkab->ki',t2,v2ijab)
  interm0 += -np.einsum('abim,ij->abjm',x2,interm1)

  '''
  C i0 ( p3 p4 h2 )_xf + = 1 * P( 2 ) * Sum ( p8 ) * x ( p3 p8 h2 )_x * i1 ( p4 p8 )_f
  C   i1 ( p3 p8 )_f + = 1 * f ( p3 p8 )_f
  C   i1 ( p3 p8 )_vt + = 1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 p3 p5 p8 )_v
  C   i1 ( p3 p8 )_vt + = 1/2 * Sum ( h6 h7 p5 ) * t ( p3 p5 h6 h7 )_t * v ( h6 h7 p5 p8 )_v
  '''
  #tmp =  np.einsum('ai,ibac->bc',t1,v2iabc)
  tmp1 =  np.einsum('ai,ian->n',t1,ChVov)
  tmp  =  np.einsum('n,bcn->bc',tmp1,ChVvv)
  tmp += -np.einsum('ibn,icn->bc',t1ChVvv,ChVov)
  interm1  =  Fock[nocc:ntot,nocc:ntot]+tmp+0.5*np.einsum('abij,ijbc->ac',t2,v2ijab)
  tmp1     =  np.einsum('abim,cb->acim',x2,interm1)
  interm0 +=  tmp1-np.einsum('acim->caim',tmp1)

  '''
  C i0 ( p3 p4 h2 )_xv + = -1 * P( 2 ) * Sum ( p8 h7 ) * x ( p3 p8 h7 )_x * i1 ( h7 p4 h2 p8 )_v
  C   i1 ( h7 p3 h1 p8 )_v + = 1 * v ( h7 p3 h1 p8 )_v
  C   i1 ( h7 p3 h1 p8 )_vt + = 1 * Sum ( p5 ) * t ( p5 h1 )_t * v ( h7 p3 p5 p8 )_v
  '''
  tmp1     = -np.einsum('abim,icjb->acjm',x2,v2iajb)
  interm0 +=  tmp1-np.einsum('acjm->cajm',tmp1)
  #interm1  =  np.einsum('ai,jbac->jbic',t1,v2iabc)
  #tmp1     = -np.einsum('acjm,jbic->abim',x2,interm1)
  #interm0 +=  tmp1-np.einsum('abim->baim',tmp1)
  tmp1 =  np.einsum('ai,jan->ijn',t1,ChVov)
  tmp2 =  np.einsum('ijn,bcn->jbic',tmp1,ChVvv)
  tmp  = -np.einsum('acjm,jbic->abim',x2,tmp2)
  tmp1 =  np.einsum('acjm,jcn->amn',x2,ChVov)
  tmp +=  np.einsum('amn,ibn->abim',tmp1,t1ChVvv)
  interm0 +=  tmp-np.einsum('abim->baim',tmp)

  '''
  C i0 ( p3 p4 h2 )_xv + = 1/2 * Sum ( p5 p6 ) * x ( p5 p6 h2 )_x * v ( p3 p4 p5 p6 )_v
  '''
  #interm0 +=  0.5*np.einsum('abim,cdab->cdim',x2,v2abcd)
  tmp1 = np.einsum('abim,acn->cbimn',x2,ChVvv)
  interm0 += np.einsum('cbimn,bdn->cdim',tmp1,ChVvv)
  

  '''
  C i0 ( p3 p4 h2 )_vxt + = -2 * P( 2 ) * Sum ( h9 ) * t ( p3 h9 )_t * i1 ( h9 p4 h2 )_vx
  C   i1 ( h9 p3 h2 )_vx + = -1/2 * Sum ( p6 ) * x ( p6 )_x * v ( h9 p3 h2 p6 )_v
  C   i1 ( h9 p3 h2 )_fx + = -1/2 * Sum ( p5 ) * x ( p3 p5 h2 )_x * i2 ( h9 p5 )_f
  C     i2 ( h9 p5 )_f + = 1 * f ( h9 p5 )_f
  C     i2 ( h9 p5 )_vt + = 1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 h9 p6 p5 )_v
  C   i1 ( h9 p3 h2 )_vx + = 1/2 * Sum ( h8 p10 ) * x ( p3 p10 h8 )_x * i2 ( h8 h9 h2 p10 )_v
  C     i2 ( h8 h9 h1 p10 )_v + = 1 * v ( h8 h9 h1 p10 )_v
  C     i2 ( h8 h9 h1 p10 )_vt + = 1 * Sum ( p5 ) * t ( p5 h1 )_t * v ( h8 h9 p5 p10 )_v
  C   i1 ( h9 p3 h2 )_vx + = 1/4 * Sum ( p6 p7 ) * x ( p6 p7 h2 )_x * v ( h9 p3 p6 p7 )_v
  C   i1 ( h9 p3 h2 )_vxt + = -1/2 * Sum ( h10 ) * t ( p3 h10 )_t * i2 ( h9 h10 h2 )_vx
  C     i2 ( h9 h10 h2 )_vx + = -1/2 * Sum ( p7 ) * x ( p7 )_x * v ( h9 h10 h2 p7 )_v
  C     i2 ( h9 h10 h2 )_vx + = 1/4 * Sum ( p7 p8 ) * x ( p7 p8 h2 )_x * v ( h9 h10 p7 p8 )_v
  C->   i2 ( h9 h10 h1 )_vxt + = 1/2 * Sum ( p5 ) * t ( p5 h1 )_t * i3 ( h9 h10 p5 )_vx
  C       i3 ( h9 h10 p5 )_vx + = 1 * Sum ( p8 ) * x ( p8 )_x * v ( h9 h10 p5 p8 )_v
  C-> i1 ( h9 p3 h1 )_vxt + = 1/2 * Sum ( p5 ) * t ( p5 h1 )_t * i2 ( h9 p3 p5 )_vx
  C     i2 ( h9 p3 p5 )_vx + = 1 * Sum ( p7 ) * x ( p7 )_x * v ( h9 p3 p5 p7 )_v
  C-> i1 ( h9 p3 h1 )_vxt + = -1/2 * Sum ( h6 p5 ) * t ( p3 p5 h1 h6 )_t * i2 ( h6 h9 p5 )_vx
  C     i2 ( h6 h9 p5 )_vx + = 1 * Sum ( p8 ) * x ( p8 )_x * v ( h6 h9 p5 p8 )_v
  '''
  interm1  = -0.5*np.einsum('bm,iajb->iajm',x1,v2iajb)
  interm1 += -0.5*np.einsum('abim,jb->jaim',x2,Ft1v2ijab)
  interm2  =  v2ijka + np.einsum('ak,ijab->ijkb',t1,v2ijab)
  interm1 +=  0.5*np.einsum('baim,ijka->jbkm',x2,interm2)
  interm1 += -0.5*np.einsum('aj,ijkm->iakm',t1,intermy)
  #>>> sign became different
  interm1 +=  0.5*np.einsum('abij,jkbm->kaim',t2,x1v2ijab)
  #<<
  tmp1     = -2*np.einsum('bi,iajm->bajm',t1,interm1)
  interm0 +=  tmp1-np.einsum('bajm->abjm',tmp1)

  #interm1  =  np.einsum('bcjm,iabc->iajm',x2,v2iabc)
  tmp1 =  x2-np.einsum('bcjm->cbjm',x2)
  tmp2 =  np.einsum('bcjm,ibn->icjmn',tmp1,ChVov)
  tmp3 =  np.einsum('icjmn,acn->iajm',tmp2,ChVvv)
  tmp  = -0.5*np.einsum('bi,iajm->bajm',t1,tmp3)
  interm0 +=  tmp-np.einsum('bajm->abjm',tmp)

  #interm2  =  np.einsum('cm,iabc->iabm',x1,v2iabc)
  #interm1  =  np.einsum('bj,iabm->iajm',t1,interm2)
  #tmp1     =  np.einsum('bi,iajm->bajm',t1,interm1)
  #interm0 +=  tmp1-np.einsum('bajm->abjm',tmp1)
  tmp2 =  np.einsum('bj,ibn->ijn',t1,ChVov)
  tmp3 =  np.einsum('bi,ijn->bjn',t1,tmp2)
  tmp  =  np.einsum('bjn,amn->bajm',tmp3,x1ChVvv)
  x1ChVov =  np.einsum('cm,icn->imn',x1,ChVov)
  tmp3 =  np.einsum('bi,imn->bmn',t1,x1ChVov)
  tmp += -np.einsum('jan,bmn->bajm',t1ChVvv,tmp3)
  interm0 +=  tmp-np.einsum('bajm->abjm',tmp)
 
  '''
  C-> i0 ( p3 p4 h1 )_vxt + = 1 * Sum ( p5 ) * t ( p5 h1 )_t * i1 ( p3 p4 p5 )_vx
  C     i1 ( p3 p4 p5 )_vx + = 1 * Sum ( p6 ) * x ( p6 )_x * v ( p3 p4 p5 p6 )_v
  '''
  #interm1  =  np.einsum('dm,abcd->abcm',x1,v2abcd)
  #interm0 += -np.einsum('ci,abcm->abim',t1,interm1)
  tmp  = -np.einsum('bmn,ian->abim',x1ChVvv,t1ChVvv)
  interm0 += tmp-np.einsum('baim->abim',tmp)

  '''
  C-> i0 ( p3 p4 h1 )_fxt + = -1 * Sum ( h5 ) * t ( p3 p4 h1 h5 )_t * i1 ( h5 )_fx
  C     i1 ( h5 )_fx + = 1 * Sum ( p9 ) * x ( p9 )_x * i2 ( h5 p9 )_f
  C       i2 ( h5 p9 )_f + = 1 * f ( h5 p9 )_f
  C       i2 ( h5 p9 )_vt + = -1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h5 h7 p6 p9 )_v
  C     i1 ( h5 )_vx + = 1/2 * Sum ( h6 p7 p8 ) * x ( p7 p8 h6 )_x * v ( h5 h6 p7 p8 )_v
  '''
  interm1  =  np.einsum('am,ia->im',x1,Ft1v2ijab)
  interm1 +=  0.5*np.einsum('ijjm->im',x2v2ijab2)
  interm0 +=  np.einsum('abij,jm->abim',t2,interm1)

  '''
  C i0 ( p3 p4 h2 )_vxt + = 1 * Sum ( h5 h6 ) * t ( p3 p4 h5 h6 )_t * i1 ( h5 h6 h2 )_vx
  C   i1 ( h5 h6 h2 )_vx + = -1/2 * Sum ( p7 ) * x ( p7 )_x * v ( h5 h6 h2 p7 )_v
  C   i1 ( h5 h6 h2 )_vx + = 1/4 * Sum ( p7 p8 ) * x ( p7 p8 h2 )_x * v ( h5 h6 p7 p8 )_v
  C-> i1 ( h5 h6 h1 )_vxt + = 1/2 * Sum ( p7 ) * t ( p7 h1 )_t * i2 ( h5 h6 p7 )_vx
  C     i2 ( h5 h6 p7 )_vx + = 1 * Sum ( p8 ) * x ( p8 )_x * v ( h5 h6 p7 p8 )_v
  '''
  interm0 +=  np.einsum('abij,ijkm->abkm',t2,intermy)

  '''
  C i0 ( p3 p4 h1 )_vxt + = 1 * P( 2 ) * Sum ( h6 p5 ) * t ( p3 p5 h1 h6 )_t * i1 ( h6 p4 p5 )_vx
  C   i1 ( h6 p3 p5 )_vx + = 1 * Sum ( p7 ) * x ( p7 )_x * v ( h6 p3 p5 p7 )_v
  C   i1 ( h6 p3 p5 )_vx + = 1 * Sum ( h7 p8 ) * x ( p3 p8 h7 )_x * v ( h6 h7 p5 p8 )_v
  '''
  tmp1     =  np.einsum('abij,jcbm->acim',t2,x2v2ijab)
  interm0 += -tmp1+np.einsum('acim->caim',tmp1)
  #interm1  =  np.einsum('dm,jcbd->jcbm',x1,v2iabc)
  #tmp1     =  np.einsum('abij,jcbm->acim',t2,interm1)
  #interm0 += -tmp1+np.einsum('acim->caim',tmp1)
  tmp2  =  np.einsum('cmn,jbn->jcbm',x1ChVvv,ChVov)
  tmp2 += -np.einsum('jmn,bcn->jcbm',x1ChVov,ChVvv)
  tmp   =  np.einsum('abij,jcbm->acim',t2,tmp2)
  interm0 += -tmp+np.einsum('acim->caim',tmp)

  return interm0

