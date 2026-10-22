"""Generates hero-tess.svg, the line mesh behind the guide cover's hero: large triangles on the
left, subdivided more often towards the right."""
import os, random
random.seed(11)
OUT = os.path.dirname(os.path.abspath(__file__))
W, H, S = 1600, 1000, 200
nx, ny = W // S, H // S
J = S * 0.3
P = {}
for j in range(ny + 1):
    for i in range(nx + 1):
        x, y = i * S, j * S
        if 0 < i < nx: x += random.uniform(-J, J)
        if 0 < j < ny: y += random.uniform(-J, J)
        P[i, j] = (x, y)

def lerp(p, q, t): return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
def d2(p, q): return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

T = []
def tri(a, b, c, depth):
    cx = (a[0] + b[0] + c[0]) / 3
    u = max(0.0, min(1.0, cx / W))
    p = 0.05 + 0.9 * u ** 1.3 - 0.12 * depth
    if depth >= 5 or random.random() > p:
        T.append((a, b, c)); return
    r = random.random()
    if r < 0.45:
        ab, bc, ca = lerp(a, b, .5), lerp(b, c, .5), lerp(c, a, .5)
        for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)): tri(*t, depth + 1)
    elif r < 0.75:
        # bisect the longest edge at an off-centre point
        a, b, c = max(((a, b, c), (b, c, a), (c, a, b)), key=lambda t: d2(t[0], t[1]))
        m = lerp(a, b, random.uniform(0.35, 0.65))
        tri(a, m, c, depth + 1); tri(m, b, c, depth + 1)
    else:
        # split around a jittered interior point
        w = [random.uniform(0.6, 1.4) for _ in range(3)]
        s = sum(w)
        m = ((a[0] * w[0] + b[0] * w[1] + c[0] * w[2]) / s, (a[1] * w[0] + b[1] * w[1] + c[1] * w[2]) / s)
        tri(a, b, m, depth + 1); tri(b, c, m, depth + 1); tri(c, a, m, depth + 1)

for j in range(ny):
    for i in range(nx):
        a, b, c, d = P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]
        if random.random() < 0.5: tri(a, b, c, 0); tri(a, c, d, 0)
        else: tri(a, b, d, 0); tri(b, c, d, 0)

f = lambda v: '%g' % round(v, 1)
E = set()
for t in T:
    for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
        E.add(tuple(sorted(((f(p[0]), f(p[1])), (f(q[0]), f(q[1]))))))
d = ''.join('M%s %sL%s %s' % (p[0], p[1], q[0], q[1]) for p, q in sorted(E))
svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 %d %d'>"
       "<path d='%s' stroke='#ffffff' stroke-opacity='.15' stroke-width='1' vector-effect='non-scaling-stroke' fill='none'/></svg>") % (W, H, d)
open(os.path.join(OUT, 'hero-tess.svg'), 'w').write(svg)
print(len(T), 'triangles,', len(svg), 'bytes')
