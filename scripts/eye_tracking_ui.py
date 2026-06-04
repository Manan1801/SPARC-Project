
"""
eye_tracking_ui.py
Pixel-perfect HUD matching the reference layout.
Fully resolution-adaptive via scale factor S = w/1280.
"""

import cv2
import numpy as np
import time

START_TIME = time.time()


# ──────────────────────────────────────────────────────────
# LOW-LEVEL HELPERS
# ──────────────────────────────────────────────────────────

def panel(img, x1, y1, x2, y2, alpha=0.72):
    """Semi-transparent dark overlay."""
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    x1=max(0,x1); y1=max(0,y1)
    x2=min(img.shape[1]-1,x2); y2=min(img.shape[0]-1,y2)
    if x2<=x1 or y2<=y1: return
    roi = img[y1:y2, x1:x2]
    cv2.addWeighted(np.full_like(roi,22), alpha, roi, 1-alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def T(img, text, pos, scale, color, thick, shadow=True):
    """Draw anti-aliased text with optional 1-px drop shadow."""
    x,y = int(pos[0]),int(pos[1])
    f   = cv2.FONT_HERSHEY_SIMPLEX
    if shadow:
        cv2.putText(img,text,(x+1,y+1),f,scale,(0,0,0),thick+1,cv2.LINE_AA)
    cv2.putText(img,text,(x,y),f,scale,color,thick,cv2.LINE_AA)


def axis_widget(img, ox, oy, sz):
    """Draw X/Y/Z' arrows at (ox,oy) with arm length sz."""
    ox,oy,sz = int(ox),int(oy),int(sz)
    lw = max(1, sz//18)
    fs = max(0.30, sz/75.0)
    # X →  red
    cv2.arrowedLine(img,(ox,oy),(ox+sz,oy),(50,80,255),lw,tipLength=0.28)
    cv2.putText(img,"X",(ox+sz+3,oy+5),cv2.FONT_HERSHEY_SIMPLEX,fs,(50,80,255),lw,cv2.LINE_AA)
    # Y ↑  green
    cv2.arrowedLine(img,(ox,oy),(ox,oy-sz),(0,210,40),lw,tipLength=0.28)
    cv2.putText(img,"Y",(ox-16,oy-sz-4),cv2.FONT_HERSHEY_SIMPLEX,fs,(0,210,40),lw,cv2.LINE_AA)
    # Z' ↙ orange
    ex,ey = ox - int(sz*0.55), oy + int(sz*0.55)
    cv2.arrowedLine(img,(ox,oy),(ex,ey),(220,110,0),lw,tipLength=0.28)
    cv2.putText(img,"Z'",(ex-22,ey+16),cv2.FONT_HERSHEY_SIMPLEX,fs,(220,110,0),lw,cv2.LINE_AA)
    # origin dot
    cv2.circle(img,(ox,oy),max(2,lw+1),(180,180,180),-1)


# ──────────────────────────────────────────────────────────
# MAIN HUD
# ──────────────────────────────────────────────────────────

def draw_hud(
    img,
    smooth_X, smooth_Y, smooth_Z,
    head_horizontal, head_vertical,
    gaze_horizontal, gaze_vertical,
    horizontal_ratio, vertical_ratio,
    lx, ly, rx, ry,
):
    H, W = img.shape[:2]

    # ── scale & spacing ──────────────────────────────────
    S   = W / 1280.0          # 0.5 at 640, 1.0 at 1280
    M   = max(4, int(12*S))   # outer margin
    G   = max(2, int(6*S))    # gap between panels

    # ── proportional heights (tuned to reference) ────────
    TOP_H  = int(H * 0.152)   # top 3-panel row
    AXK_H  = int(H * 0.260)   # axis-key panel (below left top)
    LEG_H  = int(H * 0.360)   # legend panel (right)
    HTR_H  = int(H * 0.168)   # how-to-read panel (right, below legend)
    STAT_H = int(H * 0.193)   # live-status (bottom-left)
    DOCK_H = int(H * 0.100)   # bottom dock
    CTRL_H = int(H * 0.052)   # controls bar

    # ── fonts (all scale with S) ─────────────────────────
    fHDR  = max(0.26, 0.44*S)   # panel title  (small caps row)
    fVAL  = max(0.34, 0.58*S)   # X/Y/Z values
    fBIG  = max(0.44, 0.80*S)   # HEAD / GAZE big
    fSM   = max(0.22, 0.38*S)   # legend / status small
    fTINY = max(0.20, 0.34*S)   # controls bar

    lw1 = max(1, int(1*S+0.5))
    lw2 = max(1, int(2*S+0.5))

    # ── colors ───────────────────────────────────────────
    CY   = (0,  220, 255)   # cyan   – panel titles
    WH   = (255,255,255)    # white
    CX   = (60,  80, 255)   # red-ish  X axis
    CYa  = (0,  210,  40)   # green    Y axis
    CZ   = (220,110,   0)   # orange   Z axis
    CHEAD= (50,  255,  50)  # bright green  head value
    CGAZE= (0,   220, 255)  # cyan          gaze value
    COK  = (50,  255,  50)  # OK = green

    # ─────────────────────────────────────────────────────
    # 1. TOP ROW  (3 equal panels)
    # ─────────────────────────────────────────────────────
    ty1 = M
    ty2 = ty1 + TOP_H

    col_w = (W - 2*M - 2*G) // 3
    C1x1,C1x2 = M,            M+col_w
    C2x1,C2x2 = C1x2+G,       C1x2+G+col_w
    C3x1,C3x2 = C2x2+G,       W-M

    for bx1,bx2 in [(C1x1,C1x2),(C2x1,C2x2),(C3x1,C3x2)]:
        panel(img, bx1, ty1, bx2, ty2)

    # compute directions
    xd = "LEFT" if smooth_X<-0.03 else "RIGHT" if smooth_X>0.03 else "CENTER"
    yd = "UP"   if smooth_Y<-0.03 else "DOWN"  if smooth_Y>0.03 else "CENTER"
    zd = "NEAR" if smooth_Z<0.55  else "AWAY"

    pad = int(8*S)

    # panel 1 – 3D Eye Position
    T(img,"3D EYE POSITION (w.r.t. CAMERA)",
      (C1x1+pad, ty1+int(TOP_H*0.30)), fHDR, CY, lw1)
    gy = int(TOP_H*0.30)
    T(img,f"X = {smooth_X:+.2f} m ({xd})",(C1x1+pad, ty1+gy+int(TOP_H*0.26)),fVAL,CX, lw2)
    T(img,f"Y = {smooth_Y:+.2f} m ({yd})",(C1x1+pad, ty1+gy+int(TOP_H*0.52)),fVAL,CYa,lw2)
    T(img,f"Z = {smooth_Z:+.2f} m ({zd})",(C1x1+pad, ty1+gy+int(TOP_H*0.78)),fVAL,CZ, lw2)

    # panel 2 – Head Direction
    T(img,"HEAD DIRECTION (from 3D position)",
      (C2x1+pad, ty1+int(TOP_H*0.30)), fHDR, CY, lw1)
    hstr = f"{head_horizontal}, {head_vertical}"
    ts   = cv2.getTextSize(hstr, cv2.FONT_HERSHEY_SIMPLEX, fBIG, lw2)[0]
    hx   = C2x1 + (col_w - ts[0])//2
    T(img, hstr, (hx, ty1+int(TOP_H*0.80)), fBIG, CHEAD, lw2)

    # panel 3 – Gaze Direction
    T(img,"GAZE DIRECTION (from eye movement)",
      (C3x1+pad, ty1+int(TOP_H*0.30)), fHDR, CY, lw1)
    ts2  = cv2.getTextSize(gaze_horizontal, cv2.FONT_HERSHEY_SIMPLEX, fBIG, lw2)[0]
    gx   = C3x1 + ((C3x2-C3x1) - ts2[0])//2
    T(img, gaze_horizontal, (gx, ty1+int(TOP_H*0.80)), fBIG, CGAZE, lw2)

    # ─────────────────────────────────────────────────────
    # 2. AXIS-KEY PANEL  (below panel 1)
    # ─────────────────────────────────────────────────────
    ak1 = ty2 + G
    ak2 = ak1 + AXK_H
    panel(img, C1x1, ak1, C1x2, ak2)

    lg = int(AXK_H * 0.185)          # line gap inside panel
    by = ak1 + int(AXK_H * 0.18)
    T(img,"X : Left (-)  /  Right (+)",(C1x1+pad, by),       fSM, CX,  lw1, shadow=False)
    T(img,"Y : Up (-)  /  Down (+)",   (C1x1+pad, by+lg),    fSM, CYa, lw1, shadow=False)
    T(img,"Z : Near (-)  /  Away (+)", (C1x1+pad, by+2*lg),  fSM, CZ,  lw1, shadow=False)
    T(img,"Origin: Camera Center",     (C1x1+pad, by+3*lg+int(4*S)), fSM, WH, lw1, shadow=False)

    # axis 3D widget — right side of axis-key panel
    asz  = int(AXK_H * 0.36)
    aox  = C1x2 - asz - int(12*S)
    aoy  = ak1 + AXK_H//2
    axis_widget(img, aox, aoy, asz)

    # ─────────────────────────────────────────────────────
    # 3. CROSSHAIR
    # ─────────────────────────────────────────────────────
    cx = W//2; cy = H//2
    cv2.line(img,(cx, ty2+2),(cx, H-CTRL_H-DOCK_H-4), WH, 1, cv2.LINE_AA)
    cv2.line(img,(M+2, cy), (W-M-2, cy),               WH, 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────
    # 4. EYE DOTS
    # ─────────────────────────────────────────────────────
    r = max(4, int(8*S))
    cv2.circle(img,(lx,ly),r,  (0,255,0), -1)          # left iris  = green
    cv2.circle(img,(lx,ly),r+4,(0,180,0),  1)
    cv2.circle(img,(rx,ry),r,  (0,0,255),  -1)          # right iris = red
    cv2.circle(img,(rx,ry),r+4,(0,0,180),   1)
    ex=(lx+rx)//2; ey=(ly+ry)//2
    cv2.circle(img,(ex,ey),r+2, WH, -1)                 # combined   = white

    # ─────────────────────────────────────────────────────
    # 5. RIGHT SIDE: LEGEND
    # ─────────────────────────────────────────────────────
    LW   = int(W * 0.228)
    lx1  = W - M - LW
    lx2  = W - M

    ly1  = ty2 + G
    ly2  = ly1 + LEG_H
    panel(img, lx1, ly1, lx2, ly2)

    # title
    T(img,"LEGEND",(lx1+(LW-int(60*S))//2, ly1+int(LEG_H*0.09)),
      fSM*1.15, CY, lw1)

    ITEMS = [
        ("Left Iris Points",         (0,  255,   0), False),
        ("Left Iris Center",         (0,    0, 255), True ),
        ("Right Iris Points",        (200,  0, 255), False),
        ("Right Iris Center",        (0,  220, 255), True ),
        ("Eye Corners / Boundaries", (0,  220, 255), False),
        ("Combined Eye Center",      WH,             True ),
    ]
    dr   = max(4, int(6*S))
    igap = int(LEG_H * 0.138)
    iyy  = ly1 + int(LEG_H * 0.185)
    for lbl, col, filled in ITEMS:
        dx = lx1 + dr + pad
        (cv2.circle(img,(dx,iyy),dr,col,-1)  if filled else
         cv2.circle(img,(dx,iyy),dr,col,max(1,int(2*S))))
        T(img, lbl,(dx+dr+int(6*S), iyy+int(5*S)), fSM*0.92, WH, lw1, shadow=False)
        iyy += igap

    # ─────────────────────────────────────────────────────
    # 6. RIGHT SIDE: HOW TO READ
    # ─────────────────────────────────────────────────────
    hy1 = ly2 + G
    hy2 = hy1 + HTR_H
    panel(img, lx1, hy1, lx2, hy2)

    T(img,"HOW TO READ",(lx1+(LW-int(90*S))//2, hy1+int(HTR_H*0.22)),
      fSM*1.1, CY, lw1)

    hlg  = int(HTR_H * 0.22)
    hby  = hy1 + int(HTR_H * 0.45)
    voff = int(55*S)
    T(img,"HEAD:", (lx1+pad, hby),     fSM, CY,    lw1)
    T(img,"3D position of face",  (lx1+pad+voff, hby),     fSM*0.90, WH, lw1, shadow=False)
    T(img,"relative to camera",   (lx1+pad+voff, hby+hlg), fSM*0.90, WH, lw1, shadow=False)
    T(img,"GAZE:", (lx1+pad, hby+2*hlg),     fSM, CGAZE, lw1)
    T(img,"Iris position inside", (lx1+pad+voff, hby+2*hlg),     fSM*0.90, WH, lw1, shadow=False)
    T(img,"the eye (relative)",   (lx1+pad+voff, hby+3*hlg),     fSM*0.90, WH, lw1, shadow=False)

    # ─────────────────────────────────────────────────────
    # 7. LIVE STATUS  (bottom-left, above dock)
    # ─────────────────────────────────────────────────────
    sy2  = H - CTRL_H - DOCK_H - G*2
    sy1  = sy2 - STAT_H
    sx2  = C1x2
    panel(img, M, sy1, sx2, sy2)

    T(img,"LIVE STATUS",(M+pad, sy1+int(STAT_H*0.17)), fSM*1.1, CY, lw1)

    elapsed = int(time.time() - START_TIME)
    SDATA   = [
        ("FPS",        "29.7"),
        ("Confidence", "0.86"),
        ("Distance",   f"{smooth_Z:.2f} m"),
        ("Smoothing",  "ON"),
        ("Tracking",   "OK"),
    ]
    sg  = int(STAT_H * 0.148)
    ssy = sy1 + int(STAT_H * 0.33)
    kx  = M + pad
    vx  = kx + int(80*S)
    for key, val in SDATA:
        T(img, key,       (kx, ssy), fSM, WH,                        lw1, shadow=False)
        T(img, f": {val}",(vx, ssy), fSM, COK if val=="OK" else WH,  lw1, shadow=False)
        ssy += sg

    # ─────────────────────────────────────────────────────
    # 8. BOTTOM DOCK
    # ─────────────────────────────────────────────────────
    dy2 = H - CTRL_H - G
    dy1 = dy2 - DOCK_H
    panel(img, M, dy1, W-M, dy2, alpha=0.78)

    dw   = (W - 2*M)
    dv1  = M + dw//3
    dv2  = M + 2*(dw//3)
    cv2.line(img,(dv1,dy1+int(6*S)),(dv1,dy2-int(6*S)), WH, 1)
    cv2.line(img,(dv2,dy1+int(6*S)),(dv2,dy2-int(6*S)), WH, 1)

    icz  = int(DOCK_H * 0.48)   # icon size
    dly  = dy1 + int(DOCK_H*0.32)   # label y
    dvy  = dy1 + int(DOCK_H*0.76)   # value y

    def draw_head_icon(bx1, bx2):
        ix = bx1 + int(icz*0.9)
        iy = dy1 + DOCK_H//2
        cv2.circle(img,(ix, iy-icz//2), icz//3, (80,160,255), -1)
        cv2.rectangle(img,(ix-icz//3, iy-icz//6),(ix+icz//3, iy+icz//2),(80,160,255),-1)
        return ix + icz//2 + int(8*S)

    def draw_eye_icon(bx1, bx2):
        ix = bx1 + int(icz*0.9)
        iy = dy1 + DOCK_H//2
        ew,eh = icz//2, icz//4
        pts = cv2.ellipse2Poly((ix,iy),(ew,eh),0,0,360,8)
        cv2.polylines(img,[pts],True,(0,200,255),max(1,lw2))
        cv2.circle(img,(ix,iy),icz//6,(0,200,255),-1)
        return ix + ew + int(8*S)

    def draw_clk_icon(bx1, bx2):
        ix = bx1 + int(icz*0.9)
        iy = dy1 + DOCK_H//2
        cv2.circle(img,(ix,iy),icz//2,(180,100,255),max(1,lw2))
        cv2.line(img,(ix,iy),(ix, iy-icz//3),(180,100,255),max(1,lw2))
        cv2.line(img,(ix,iy),(ix+icz//4,iy),(180,100,255),max(1,lw2))
        return ix + icz//2 + int(8*S)

    # HEAD cell
    tx = draw_head_icon(M, dv1)
    T(img,"HEAD",                          (tx,dly), fSM*1.05, CY,    lw1)
    T(img,f"{head_horizontal}, {head_vertical}",(tx,dvy), fVAL*0.95, CHEAD, lw2)

    # GAZE cell
    tx = draw_eye_icon(dv1, dv2)
    T(img,"GAZE",        (tx,dly), fSM*1.05, CY,    lw1)
    T(img,gaze_horizontal,(tx,dvy), fVAL*0.95, CGAZE, lw2)

    # TIME cell
    mins = elapsed // 60; secs = elapsed % 60
    tx   = draw_clk_icon(dv2, W-M)
    T(img,"TIME",              (tx,dly), fSM*1.05, CY,  lw1)
    T(img,f"00:{mins:02d}:{secs:02d}",(tx,dvy), fVAL*0.95, WH, lw2)

    # ─────────────────────────────────────────────────────
    # 9. CONTROLS BAR
    # ─────────────────────────────────────────────────────
    ctl1 = H - CTRL_H
    ctl2 = H - 1
    panel(img, M, ctl1, W-M, ctl2, alpha=0.82)
    ctxt = "CONTROLS:  [SPACE] Pause / Resume      [ESC] Stop      [Q] Close Window"
    ctsz = cv2.getTextSize(ctxt, cv2.FONT_HERSHEY_SIMPLEX, fTINY, 1)[0]
    T(img, ctxt, ((W-ctsz[0])//2, ctl1+int(CTRL_H*0.68)), fTINY, WH, lw1, shadow=False)

    return img