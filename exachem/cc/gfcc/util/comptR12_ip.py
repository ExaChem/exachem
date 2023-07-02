def __ComptR1_IP__(nocc,ntot,x1,x2,x3,t1,t2,Fock,\
                   v2ijab,v2ijka,\
		   ChVov,ChVoo,ChVvv):
  '''
  Compute R1 contribution, i.e. x1^T ( Hss x1 + Hsd x2 )
  '''
  import numpy as np
  import numpy.linalg as npla

  Ft1v2ijab =  Fock[0:nocc,nocc:ntot] + np.einsum('ai,ijab->jb',t1,v2ijab) 
  '''
  C i0 ( h1 )_xf + = -1 * Sum ( h6 ) * x ( h6 )_x * i1 ( h6 h1 )_f
  C   i1 ( h6 h1 )_f + = 1 * f ( h6 h1 )_f
  C   i1 ( h6 h1 )_ft + = 1 * Sum ( p7 ) * t ( p7 h1 )_t * i2 ( h6 p7 )_f
  C     i2 ( h6 p7 )_f + = 1 * f ( h6 p7 )_f
  C     i2 ( h6 p7 )_vt + = 1 * Sum ( h5 p4 ) * t ( p4 h5 )_t * v ( h5 h6 p4 p7 )_v
  C   i1 ( h6 h1 )_vt + = -1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 h6 h1 p3 )_v
  C   i1 ( h6 h1 )_vt + = -1/2 * Sum ( h5 p3 p4 ) * t ( p3 p4 h1 h5 )_t * v ( h5 h6 p3 p4 )_v
  '''
  interm1  =  Fock[0:nocc,0:nocc] + np.einsum('ai,ja->ji',t1,Ft1v2ijab)
  interm0  = -np.einsum('im,ij->jm',x1,interm1)
  interm0 +=  np.einsum('jm,ai,ijka->km',x1,t1,v2ijka)
  tmp  =  0.5*np.einsum('km,jkab->jabm',x1,v2ijab)
  interm0 +=  np.einsum('abij,jabm->im',t2,tmp)

  '''
  C i0 ( h1 )_xf + = -1 * Sum ( p7 h6 ) * x ( p7 h1 h6 )_x * i1 ( h6 p7 )_f
  C   i1 ( h6 p7 )_f + = 1 * f ( h6 p7 )_f
  C   i1 ( h6 p7 )_vt + = 1 * Sum ( h4 p3 ) * t ( p3 h4 )_t * v ( h4 h6 p3 p7 )_v
  '''
  interm0 += -np.einsum('bijm,jb->im',x2,Ft1v2ijab)
 
  '''
  C i0 ( h1 )_xv + = 1/2 * Sum ( p7 h6 h8 ) * xp ( p7 h6 h8 )_x * i1 ( h6 h8 h1 p7 )_v
  C   i1 ( h6 h8 h1 p7 )_v + = 1.0 * v ( h6 h8 h1 p7 )_v
  C   i1 ( h6 h8 h1 p7 )_vt + = 1.0 * Sum ( p3 ) * t ( p3 h1 )_t * v ( h6 h8 p3 p7 )_v
  '''
  interm1  =  v2ijka + np.einsum('ai,jkab->jkib',t1,v2ijab)
  interm0 +=  0.5*np.einsum('bijm,ijkb->km',x2,interm1)

  return interm0


def __ComptR2_IP__(nocc,ntot,x1,x2,x3,t1,t2,Fock,\
                   v2ijab,v2ijka,v2iajb,\
                   v2ijkl,\
		   ChVov,ChVoo,ChVvv):
  '''
  Compute R2 contribution, i.e. x2^T ( Hds x1 + Hdd x2 )
  Note: x (dummy, hx), x ( px dummy hx hx ), and interm0 ( px dummy hx hx ) 
  '''
  import numpy as np
  import numpy.linalg as npla

  Ft1v2ijab  =  Fock[0:nocc,nocc:ntot] + np.einsum('ai,ijab->jb',t1,v2ijab)
  t1v2ijab   =  np.einsum('ai,jkba->jkib',t1,v2ijab)
  t1ChVov    =  np.einsum('ak,jan->kjn',t1,ChVov)
  t1ChVvv    =  np.einsum('ai,abn->ibn',t1,ChVvv)
  x1ChVov    =  np.einsum('km,kan->amn',x1,ChVov) 
  x2v2ijab   =  np.einsum('cklm,ilac->ikam',x2,v2ijab)
  tmp  =  np.einsum('ai,jkla->jkli',t1,v2ijka-0.5*t1v2ijab)
  intermx    =  v2ijkl+tmp-np.einsum('jkli->jkil',tmp)
  intermy    =  v2ijka-t1v2ijab
  '''
  C i0 ( p4 h1 h2 )_xv + = Sum ( h9 ) * x ( h9 )_x * i1 ( h9 p4 h1 h2 )_v
  C   i1 ( h9 p3 h1 h2 )_v + = 1 * v ( h9 p3 h1 h2 )_v
  C   i1 ( h9 p3 h1 h2 )_vt + = -1 * P( 2 ) * Sum ( p5 ) * t ( p5 h1 )_t * i2 ( h9 p3 h2 p5 )_v
  C     i2 ( h9 p3 h1 p5 )_v + = 1 * v ( h9 p3 h1 p5 )_v
  C     i2 ( h9 p3 h1 p5 )_vt + = -1/2 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h9 p3 p5 p6 )_v
  C   i1 ( h9 p3 h1 h2 )_ft + = -1 * Sum ( p8 ) * t ( p3 p8 h1 h2 )_t * i2 ( h9 p8 )_f
  C     i2 ( h9 p8 )_f + = 1 * f ( h9 p8 )_f
  C     i2 ( h9 p8 )_vt + = 1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 h9 p6 p8 )_v
  C   i1 ( h9 p3 h1 h2 )_vt + = 1 * P( 2 ) * Sum ( h6 p5 ) * t ( p3 p5 h1 h6 )_t * i2 ( h6 h9 h2 p5 )_v
  C     i2 ( h6 h9 h1 p5 )_v + = 1 * v ( h6 h9 h1 p5 )_v
  C     i2 ( h6 h9 h1 p5 )_vt + = -1 * Sum ( p7 ) * t ( p7 h1 )_t * v ( h6 h9 p5 p7 )_v
  C   i1 ( h9 p3 h1 h2 )_vt + = 1/2 * Sum ( p5 p6 ) * t ( p5 p6 h1 h2 )_t * v ( h9 p3 p5 p6 )_v
  '''
#
#  interm2  = -0.5*np.einsum('ai,jbca->jbic',t1,v2iabc)
  tmp1 =  np.einsum('jm,ijn->imn',x1,t1ChVov)
  tmp  =  np.einsum('kbn,imn->bikm',t1ChVvv,tmp1)
  interm0  =  tmp-np.einsum('bikm->bkim',tmp)
#
  tmp1 =  np.einsum('jm,jbka->bkam',x1,v2iajb)
  tmp  =  np.einsum('ai,bkam->bikm',t1,tmp1)
  interm0 +=  np.einsum('km,ijka->aijm',x1,v2ijka)\
             -tmp+np.einsum('bikm->bkim',tmp)
  tmp  =  np.einsum('km,kb->bm',x1,Ft1v2ijab)
  interm0 += -np.einsum('abij,bm->aijm',t2,tmp)
#
  tmp1 =  np.einsum('km,jklb->jlbm',x1,intermy)
  tmp  =  np.einsum('abij,jlbm->ailm',t2,tmp1)
  interm0 +=  tmp-np.einsum('ailm->alim',tmp)
#
#  tmpy  =  0.5*np.einsum('km,kcab->cabm',x1,v2iabc)
#  tmpx +=  np.einsum('abij,cabm->cijm',t2,tmpy)
#
  tmp  =  np.einsum('amn,bcn->cabm',x1ChVov,ChVvv)
  interm0 += np.einsum('abij,cabm->cijm',t2,tmp)

  '''
  C i0 ( p3 h1 h2 )_xf + = -1 * P( 2 ) * Sum ( h8 ) * x ( p3 h1 h8 )_x * i1 ( h8 h2 )_f 
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
  tmp  =  np.einsum('aijm,jk->aikm',x2,interm1) 
  interm0 += -tmp+np.einsum('aikm->akim',tmp)

  '''
  C i0 ( p4 h1 h2 )_xf + = 1 * Sum ( p8 ) * x ( p8 h1 h2 )_x * i1 ( p4 p8 )_f
  C   i1 ( p3 p8 )_f + = 1 * f ( p3 p8 )_f
  C   i1 ( p3 p8 )_vt + = 1 * Sum ( h6 p5 ) * t ( p5 h6 )_t * v ( h6 p3 p5 p8 )_v
  C   i1 ( p3 p8 )_vt + = 1/2 * Sum ( h6 h7 p5 ) * t ( p3 p5 h6 h7 )_t * v ( h6 h7 p5 p8 )_v
  '''
#  interm1 +=  np.einsum('ai,ibac->bc',t1,v2iabc)
  tmp = np.einsum('iin->n',t1ChVov)
  interm1  =  Fock[nocc:ntot,nocc:ntot]+np.einsum('n,bcn->bc',tmp,ChVvv)
  interm1 += -np.einsum('ibn,icn->bc',t1ChVvv,ChVov) 
#
  interm1 +=  0.5*np.einsum('abij,ijbc->ac',t2,v2ijab)
  interm0 +=  np.einsum('bijm,cb->cijm',x2,interm1)

  '''
  C i0 ( p3 h1 h2 )_xv + = 1/2 * Sum ( h9 h10 ) * x ( p3 h9 h10 )_x * i1 ( h9 h10 h1 h2 )_v 
  C   i1 ( h9 h10 h1 h2 )_v + = 1 * v ( h9 h10 h1 h2 )_v
  C   i1 ( h9 h10 h1 h2 )_vt + = -1 * P( 2 ) * Sum ( p5 ) * t ( p5 h1 )_t * i2 ( h9 h10 h2 p5 )_v
  C     i2 ( h9 h10 h1 p5 )_v + = 1 * v ( h9 h10 h1 p5 )_v
  C     i2 ( h9 h10 h1 p5 )_vt + = -1/2 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h9 h10 p5 p6 )_v
  C   i1 ( h9 h10 h1 h2 )_vt + = 1/2 * Sum ( p5 p6 ) * t ( p5 p6 h1 h2 )_t * v ( h9 h10 p5 p6 )_v
  '''
  tmp   =  np.einsum('cklm,klab->abcm',x2,v2ijab)
  interm0 +=  0.5*np.einsum('aijm,ijkl->aklm',x2,intermx)\
             +0.25*np.einsum('abij,abcm->cijm',t2,tmp)

  '''
  C i0 ( p4 h1 h2 )_xv + = -1 * P( 2 ) * Sum ( p8 h7 ) * x ( p8 h1 h7 )_x * i1 ( h7 p4 h2 p8 )_v
  C   i1 ( h7 p3 h1 p8 )_v + = 1 * v ( h7 p3 h1 p8 )_v
  C   i1 ( h7 p3 h1 p8 )_vt + = 1 * Sum ( p5 ) * t ( p5 h1 )_t * v ( h7 p3 p5 p8 )_v
  '''
#  interm1 +=  np.einsum('ak,jcab->jckb',t1,v2iabc)
  interm1  =  v2iajb+np.einsum('kjn,bcn->jckb',t1ChVov,ChVvv)
  interm1 += -np.einsum('kcn,jbn->jckb',t1ChVvv,ChVov)
#
  tmp  =  np.einsum('bijm,jckb->cikm',x2,interm1)
  interm0 += -tmp+np.copy(np.einsum('cikm->ckim',tmp))

  '''
  C i0 ( p3 h1 h2 )_vxt + = 1 * Sum ( h10 ) * t ( p3 h10 )_t * i1 ( h10 h1 h2 )_vx
  C   i1 ( h10 h1 h2 )_vx + = -1 * Sum ( h8 ) * x ( h8 )_x * i2 ( h8 h10 h1 h2 )_v
  -C     i2 ( h8 h10 h1 h2 )_v + = 1 * v ( h8 h10 h1 h2 )_v
  C     i2 ( h8 h10 h1 h2 )_vt + = -1 * P( 2 ) * Sum ( p5 ) * t ( p5 h1 )_t * i3 ( h8 h10 h2 p5 )_v
  -C       i3 ( h8 h10 h1 p5 )_v + = 1 * v ( h8 h10 h1 p5 )_v
  C       i3 ( h8 h10 h1 p5 )_vt + = -1/2 * Sum ( p6 ) * t ( p6 h1 )_t * v ( h8 h10 p5 p6 )_v
  -C     i2 ( h8 h10 h1 h2 )_vt + = 1/2 * Sum ( p5 p6 ) * t ( p5 p6 h1 h2 )_t * v ( h8 h10 p5 p6 )_v
  -C   i1 ( h10 h1 h2 )_fx + = 1 * Sum ( p5 ) * x ( p5 h1 h2 )_x * i2 ( h10 p5 )_f
  -C     i2 ( h10 p5 )_f + = 1 * f ( h10 p5 )_f
  -C     i2 ( h10 p5 )_vt + = -1 * Sum ( h7 p6 ) * t ( p6 h7 )_t * v ( h7 h10 p5 p6 )_v
  -C   i1 ( h10 h1 h2 )_vx + = -1 * P( 2 ) * Sum ( h8 p9 ) * x ( p9 h1 h8 )_x * i2 ( h8 h10 h2 p9 )_v
  -C     i2 ( h8 h10 h1 p9 )_v + = 1 * v ( h8 h10 h1 p9 )_v
  -C     i2 ( h8 h10 h1 p9 )_vt + = 1 * Sum ( p5 ) * t ( p5 h1 )_t * v ( h8 h10 p5 p9 )_v
  '''
  interm1  = -np.einsum('im,ijkl->jklm',x1,intermx)
  interm2  =  0.5*np.einsum('km,klab->labm',x1,v2ijab)
  interm1 += -np.einsum('abij,labm->lijm',t2,interm2)
  interm1 += -np.einsum('bijm,kb->kijm',x2,Ft1v2ijab)
  interm2  =  np.einsum('bijm,jklb->klim',x2,intermy)
  interm1 += -interm2+np.copy(np.einsum('klim->kilm',interm2))
  interm0 +=  np.einsum('ai,iklm->aklm',t1,interm1)

  '''
  C i0 ( p3 h1 h2 )_vxt + = 1/2 * Sum ( p5 ) * t ( p3 p5 h1 h2 )_t * i1 ( p5 )_vx
  C   i1 ( p5 )_vx + = -1 * Sum ( h6 h7 p8 ) * x ( p8 h6 h7 )_x * v ( h6 h7 p5 p8 )_v
  '''
  tmp  =  np.einsum('iibm->bm',x2v2ijab)
  interm0 +=  0.5*np.einsum('abij,bm->aijm',t2,tmp)

  '''
  C i0 ( p3 h1 h2 )_vxt + = 1 * P( 2 ) * Sum ( h6 p5 ) * t ( p3 p5 h1 h6 )_t * i1 ( h6 h2 p5 )_vx
  C   i1 ( h6 h1 p5 )_vx + = 1 * Sum ( h7 p8 ) * x ( p8 h1 h7 )_x * v ( h6 h7 p5 p8 )_v
  '''
  tmp  =  np.einsum('abij,jkbm->akim',t2,x2v2ijab)
  interm0 += tmp-np.einsum('akim->aikm',tmp)
  
  return interm0
