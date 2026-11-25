"""Editable vector schematic following the supplied Figure 1 sketch."""
from pathlib import Path
import os
import sys
import json
import math
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE/'cache/matplotlib')
os.environ['XDG_CACHE_HOME'] = str(HERE/'cache')
sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Circle, Polygon, Rectangle, Wedge, FancyArrowPatch
from matplotlib.transforms import Affine2D

sys.path.insert(0, '/Users/junheqiao/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment

plt.rcParams.update({
    'font.family':'sans-serif', 'font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
    'font.size':8, 'svg.fonttype':'none', 'pdf.fonttype':42,
    'mathtext.fontset':'custom', 'mathtext.rm':'Arial',
    'mathtext.it':'Arial:italic', 'mathtext.bf':'Arial:bold',
    'axes.unicode_minus':True, 'savefig.facecolor':'white',
})
BLUE='#24668C'; BLUE_LIGHT='#EAF4F9'; AMBER='#AE701D'
INK='#243745'; MUTED='#667780'; LINE='#8DA6B4'; BEIGE='#F7F0E1'
FISHES=[]


class Fish(Artist):
    """One independently editable SVG group, with the nose pointing along angle."""
    def __init__(self, ax, x, y, angle=0, length=5.6, color=BLUE, name='fish'):
        super().__init__()
        self.set_gid(name)
        self.set_zorder(5)
        tr=Affine2D().scale(length).rotate_deg(angle).translate(x,y)+ax.transData
        # Rounded head, elongated body, narrow caudal peduncle; head is +x.
        vertices=[(-.36,0),(-.26,.08),(-.08,.145),(.16,.125),
                  (.29,.11),(.43,.065),(.50,.015),(.52,-.012),
                  (.36,-.071),(.20,-.099),(-.10,-.12),(-.28,-.062),
                  (-.36,0),(-.36,0)]
        codes=[MPath.MOVETO]+[MPath.CURVE4]*12+[MPath.CLOSEPOLY]
        self.parts=[
            Polygon([(-.34,0),(-.57,.16),(-.52,.015),(-.57,-.15),(-.32,0)],
                    closed=True,facecolor=color,edgecolor=color,lw=.25),
            Polygon([(-.22,.069),(-.18,.205),(-.02,.13),(.09,.115)],
                    closed=True,facecolor=color,edgecolor=color,lw=.2),
            Polygon([(-.16,-.07),(-.11,-.19),(.05,-.11)],
                    closed=True,facecolor=color,edgecolor=color,lw=.2),
            PathPatch(MPath(vertices,codes),facecolor=color,edgecolor=color,lw=.3),
            Circle((.363,.035),.027,facecolor='white',edgecolor='none'),
            PathPatch(MPath([(.24,.083),(.19,.025),(.22,-.065)],
                            [MPath.MOVETO,MPath.CURVE3,MPath.CURVE3]),
                      facecolor='none',edgecolor='white',lw=.28,alpha=.8),
        ]
        for p in self.parts:
            p.set_transform(tr);p.set_figure(ax.figure)
        FISHES.append(dict(id=name,x=x,y=y,angle_deg=angle,length_mm=length,color=color))
    def draw(self, renderer):
        renderer.open_group('fish',gid=self.get_gid())
        for p in self.parts:p.draw(renderer)
        renderer.close_group('fish')


def fish(ax,x,y,angle=0,length=5.6,color=BLUE,name='fish'):
    f=Fish(ax,x,y,angle,length,color,name);ax.add_artist(f);return f


def label(ax,x,y,s,**kw):
    base=dict(ha='center',va='center',fontsize=8,color=INK,zorder=9)
    base.update(kw)
    return ax.text(x,y,s,**base)


def arrow(ax,start,end,color=MUTED,lw=.65,style='-|>',rad=0,scale=6,z=4):
    p=FancyArrowPatch(start,end,arrowstyle=style,mutation_scale=scale,
        linewidth=lw,color=color,connectionstyle=f'arc3,rad={rad}',
        shrinkA=0,shrinkB=0,zorder=z)
    ax.add_patch(p)
    return p


def trajectory(ax,coords):
    codes=[MPath.MOVETO]+[MPath.CURVE4]*(len(coords)-1)
    patch=PathPatch(MPath(coords,codes),fc='none',ec=LINE,lw=.8,
                    linestyle=(0,(3,3)),zorder=2)
    ax.add_patch(patch)


def shoal(ax,center,offsets,angle,color,name,length=5.4):
    for j,(dx,dy,da) in enumerate(offsets,1):
        fish(ax,center[0]+dx,center[1]+dy,angle+da,length,color,f'{name}-{j:02}')


def main():
    fig=plt.figure(figsize=(180/25.4,153/25.4))
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,180),ylim=(0,153),aspect='equal');ax.axis('off')
    ax.add_patch(Rectangle((17,12),157,107,facecolor='#FAFCFD',edgecolor='none',zorder=0))
    ax.add_patch(Rectangle((17,102),157,17,facecolor=BLUE_LIGHT,edgecolor='none',zorder=0))
    ax.add_patch(Rectangle((17,12),157,27,facecolor=BEIGE,edgecolor='none',zorder=0))
    ax.plot([17,174],[102,102],color='#B0C8D6',lw=.7,zorder=1)
    ax.plot([17,174],[39,39],color='#CCB987',lw=.7,zorder=1)
    ax.plot([17,174,174],[12,12,119],color='#6A818D',lw=.8,zorder=1)
    ax.plot([17,174],[119,119],color=BLUE,lw=1.1,zorder=6)
    # Depth increases downwards; use the requested h_b boundary label.
    arrow(ax,(17,123),(17,7.5),color=INK,lw=.8,scale=7,z=7)
    label(ax,17,4.6,r'$h$',fontsize=9)
    for y,tag,color in [(119,r'$h=0$',BLUE),(102,r'$h_d$',BLUE),
                        (39,r'$h_b$',AMBER),(12,r'$L_z$',INK)]:
        ax.plot([15.8,17],[y,y],color=INK,lw=.65,zorder=7)
        label(ax,14.6,y,tag,ha='right',color=color,fontsize=8)
    label(ax,20,122.4,'Water surface',ha='left',color=BLUE,fontsize=7.6)
    label(ax,20,105.4,'Decision band',ha='left',color=BLUE,fontsize=7.6)
    label(ax,20,42.4,'Rest zone',ha='left',color=AMBER,fontsize=7.6)

    # Dashed paths convey the cycle; all direction-bearing shoal marks are fish.
    trajectory(ax,[(43,26),(58,34),(66,53),(82,67),
                   (97,82),(106,97),(114,108)])
    trajectory(ax,[(134,112),(134,96),(136,64),(137,44),
                   (137,31),(141,29),(147,29),
                   (158,31),(160,69),(166,106)])

    horizontal=[(-6,4,3),(1,4,-3),(8,4,4),(-8,0,-2),(-1,0,2),(6,0,0),
                (-5,-4,2),(2,-4,-2),(9,-4,2)]
    diagonal=[(-5,3,-3),(1,4,2),(7,5,-2),(-7,-1,2),(-1,0,-3),(5,1,2),
              (-4,-5,0),(2,-4,2)]
    shoal(ax,(34,25.5),horizontal,0,BLUE,'stay-rest',length=5.2)
    label(ax,35,16.3,r'Stay $\zeta_s$',color=BLUE,fontsize=8)
    shoal(ax,(58,43.5),diagonal,45,BLUE,'stay-liftoff',length=5.1)
    label(ax,58,31.1,'Lift-off',fontsize=8.2)
    rise_offsets=[(-5,8,-3),(1,4,2),(7,9,-2),(-7,-1,2),(-1,0,-3),(5,1,2),(-4,-5,0),(2,-4,2)]
    shoal(ax,(82,69),rise_offsets,45,BLUE,'stay-rise',length=5.2)
    label(ax,82,59.8,r'Stay $\zeta_s$',color=BLUE,fontsize=8)
    shoal(ax,(106,94),[(-4,3,0),(1,4,0),(-6,-1,0),(-1,0,0),(4,1,0)],
          50,BLUE,'stay-near-surface',length=4.8)
    # One accepted breather defines the first release; small circles are air bubbles.
    fish(ax,114,106,67,5.6,AMBER,'leave-first-release')
    for x,y,r in [(114.8,109.9,.43),(116,111,.56),(113.8,111.8,.35)]:
        ax.add_patch(Circle((x,y),r,fc='white',ec=AMBER,lw=.55,zorder=7))
    label(ax,122.2,98.7,r'Leave $\zeta_l$',color=AMBER,fontsize=8)
    label(ax,89.8,110.3,'First air release',fontsize=7.8)
    arrow(ax,(101.5,109.8),(111.4,107.6),color=MUTED,lw=.65,scale=5.4)

    shoal(ax,(121.2,116.65),[(0,0,0)],
          90,AMBER,'accumulation',length=4.7)
    label(ax,121.5,110,r'$\zeta_a$',color=AMBER,fontsize=8.6)
    label(ax,123,125.3,'Accumulation',color=AMBER,fontsize=7.6)
    arrow(ax,(123,122.5),(121.2,119),color=AMBER,lw=.6,scale=5)
    # Earlier breathers are already deeper than the last fish to exit.
    shoal(ax,(134,112.6),[(-2.7,-6.4,25),(0,-8.6,0),(3,-1,-15)],
          -90,AMBER,'turn',length=4.8)
    label(ax,143,109.2,r'Turn $\zeta_t$',color=AMBER,ha='left',fontsize=8)
    arrow(ax,(128.5,118.1),(134,116.2),color=AMBER,lw=.75,rad=-.65,scale=5.5)
    label(ax,145.8,125.3,'Last exit',fontsize=7.8)
    arrow(ax,(142.5,122.3),(137.2,111.5),color=MUTED,lw=.65,scale=5.4)

    shoal(ax,(136,83),[(-4,4,0),(0,5,0),(4,4,0),(-3,-2,0),(1,-1,0),(5,-2,0)],
          -90,AMBER,'back-descent',length=5.4)
    label(ax,145,83,r'Back $\zeta_b$',color=AMBER,ha='left',fontsize=8)
    shoal(ax,(145,29.5),[(-4,3,0),(2,3,0),(-7,-1,0),(-1,-1,0),(5,-1,0),(-3,-5,0),(3,-5,0)],
          0,BLUE,'stay-return',length=5.1)
    label(ax,145,19,r'Stay $\zeta_s$',color=BLUE,fontsize=8)
    fish(ax,166,106,70,5.6,AMBER,'leave-next-event')
    for x,y,r in [(166.8,109.8,.43),(167.7,111.2,.35)]:
        ax.add_patch(Circle((x,y),r,fc='white',ec=AMBER,lw=.55,zorder=7))
    label(ax,154,97,'Next event',fontsize=7.8)

    # Interaction-zone detail. Repulsion is omnidirectional; the 25-degree
    # rear blind sector applies to orientation and attraction only.
    cx,cy=40,79
    ax.add_patch(Wedge((cx,cy),13,-167.5,167.5,facecolor='#ECF3F7',edgecolor='#5F879F',lw=.7,zorder=2))
    ax.add_patch(Wedge((cx,cy),9,-167.5,167.5,facecolor='#D9E8F1',edgecolor='#5F879F',lw=.7,zorder=2))
    ax.add_patch(Circle((cx,cy),4.6,fc='#BCD3E2',ec='#5F879F',lw=.7,zorder=3))
    fish(ax,cx,cy-.8,0,6.8,BLUE,'interaction-focal')
    label(ax,cx,cy+2.1,r'$r_r$',fontsize=7.8)
    label(ax,cx,cy+6.5,r'$r_o$',fontsize=7.8)
    label(ax,cx,cy+10.7,r'$r_a$',fontsize=7.8)
    for angle in [167.5,192.5]:
        th=np.deg2rad(angle)
        ax.plot([cx+4.6*np.cos(th),cx+16*np.cos(th)],
                [cy+4.6*np.sin(th),cy+16*np.sin(th)],color=MUTED,lw=.6,zorder=3)
    # alpha_f spans the visible 335-degree sector, not the rear blind wedge.
    fov_arc=MPath.arc(-167.5,167.5).transformed(
        Affine2D().scale(14.7).translate(cx,cy))
    ax.add_patch(FancyArrowPatch(path=fov_arc,arrowstyle='<->',
        mutation_scale=4.5,lw=.65,color=MUTED,zorder=4))
    label(ax,40,60.7,r'Field of view $\alpha_f$',fontsize=7.8)
    arrow(ax,(40,62.8),(40,64.3),style='-',lw=.6)
    # These callout arrows magnify two fish; they are explanatory, not fish.
    ax.add_patch(Circle((77,77),3.6,fc='none',ec=MUTED,lw=.6,zorder=8))
    arrow(ax,(73.3,77.5),(57,79),color=MUTED,lw=.65,scale=6)
    ax.add_patch(Circle((89,78),3.6,fc='none',ec=MUTED,lw=.6,zorder=8))
    arrow(ax,(92.7,77.5),(106.8,65.1),color=MUTED,lw=.65,scale=6)

    # Vertical force vectors retain arrowheads and are visually distinct from fish.
    fish(ax,112,63,40,7.3,BLUE,'force-focal')
    arrow(ax,(112,63),(112,70),color=AMBER,lw=1,scale=7.5,z=7)
    arrow(ax,(112,63),(112,56),color=BLUE,lw=1,scale=7.5,z=7)
    label(ax,116,70,r'$F_{\tau}$',color=AMBER,ha='left',fontsize=9)
    label(ax,120.5,70,'Oxygen\ndrive',color=AMBER,ha='left',fontsize=7.4)
    label(ax,116,56,r'$F_s$',color=BLUE,ha='left',fontsize=9)
    label(ax,120.5,56,'Surface\navoidance',color=BLUE,ha='left',fontsize=7.4)

    # Event endpoints align to the schematic stages; spacing does not encode time.
    for x,y0,tag in [(114,112.8,r'$B_m$'),(137,108.9,r'$E_m$'),(166,112.8,r'$B_{m+1}$')]:
        ax.plot([x,x],[y0,127.9],color='#9BA6AC',lw=.55,ls=(0,(2.5,3)),zorder=1)
        ax.plot([x,x],[134.1,142],color='#9BA6AC',lw=.55,ls=(0,(2.5,3)),zorder=1)
        label(ax,x,131,tag,fontsize=9.2)
    arrow(ax,(114,140),(137,140),style='<->',lw=.7,scale=6)
    arrow(ax,(137,140),(166,140),style='<->',lw=.7,scale=6)
    label(ax,125.5,146,r'$T_m$',fontsize=9.5)
    label(ax,151.5,146,r'$I_m$',fontsize=9.5)

    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=HERE/'qa/alignment.json',
                                      overlay_svg=HERE/'qa/alignment.svg',strict=True)
    stem=HERE/'figure1_sketch'
    fig.savefig(stem.with_suffix('.svg'))
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.png'),dpi=600)
    # Helpful descriptions for vector editors, with no raster dependency.
    svg=stem.with_suffix('.svg');tree=ET.parse(svg);root=tree.getroot()
    ns='http://www.w3.org/2000/svg'
    ET.register_namespace('',ns);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
    desc=ET.Element('{'+ns+'}desc')
    desc.text='Spatial schooling and breathing-cycle schematic. Named fish groups are editable vector paths; text remains editable. Fish noses indicate heading. Thin arrows are explanatory callouts, force vectors, or event-time intervals.'
    root.insert(0,desc);tree.write(svg,encoding='utf-8',xml_declaration=True)
    (HERE/'qa/fish_manifest.json').write_text(json.dumps(FISHES,indent=2)+'\n')
    (HERE/'qa/export_manifest.json').write_text(json.dumps(dict(width_mm=180,height_mm=153,
        fish_count=len(FISHES),raster_images_in_svg=len(root.findall('.//{'+ns+'}image')),
        editable_text_elements=len(root.findall('.//{'+ns+'}text')),states=['stay','leave','accumulation','turn','back']),indent=2)+'\n')
    plt.close(fig)
    print(f'Exported {stem.name}: SVG, PDF, 600 dpi PNG; {len(FISHES)} grouped fish.')


if __name__=='__main__':main()
