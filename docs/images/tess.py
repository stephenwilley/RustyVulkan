"""Generates tess-light.svg and tess-dark.svg, the chapter-card backgrounds on the guide cover."""
import os, random
OUT = os.path.dirname(os.path.abspath(__file__))
random.seed(7)
W,H,S=960,480,60
nx,ny=W//S,H//S
J=S*0.28
P={}
for j in range(ny+1):
  for i in range(nx+1):
    x,y=i*S,j*S
    if 0<i<nx: x+=random.uniform(-J,J)
    if 0<j<ny: y+=random.uniform(-J,J)
    P[i,j]=(x,y)
T=[]
def tri(a,b,c,d):
  if d==0: T.append((a,b,c)); return
  m=lambda p,q:((p[0]+q[0])/2,(p[1]+q[1])/2)
  ab,bc,ca=m(a,b),m(b,c),m(c,a)
  for t in ((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)): tri(*t,d-1)
for j in range(ny):
  for i in range(nx):
    a,b,c,d=P[i,j],P[i+1,j],P[i+1,j+1],P[i,j+1]
    lv=lambda: 1 if random.random()<0.22 else 0
    if random.random()<0.5: tri(a,b,c,lv()); tri(a,c,d,lv())
    else: tri(a,b,d,lv()); tri(b,c,d,lv())
# weights: base colour most common
W8=[4,2,2,2,1,1,1]
choice=[random.choices(range(7),W8)[0] for _ in T]
pal={'light':['#f6f6f6','#f9f9f9','#fbfbfb','#f1f1f1','#ededed','#f8eef0','#f5e8eb'],
     'dark': ['#1f1f1f','#232323','#262626','#1b1b1b','#181818','#251d1f','#2a1e22']}
f=lambda v:'%g'%round(v)
for name,cols in pal.items():
  paths=[]
  for k,c in enumerate(cols):
    d=''.join('M%s %sL%s %sL%s %sZ'%(f(a[0]),f(a[1]),f(b[0]),f(b[1]),f(cc[0]),f(cc[1])) for (a,b,cc),ch in zip(T,choice) if ch==k)
    paths.append("<path fill='%s' stroke='%s' stroke-width='.75' stroke-linejoin='round' d='%s'/>"%(c,c,d))
  svg="<svg xmlns='http://www.w3.org/2000/svg' width='%d' height='%d'>%s</svg>"%(W,H,''.join(paths))
  open(OUT+'/tess-%s.svg'%name,'w').write(svg)
  print(name,len(svg))
