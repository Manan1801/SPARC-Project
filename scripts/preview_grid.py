#!/usr/bin/env python3
# preview_grid.py (final polish)
import time
from collections import deque
import numpy as np
import cv2
from tunables import DEFAULT_CELL_W, DEFAULT_CELL_H, PLOT_UPDATE_INTERVAL

class PreviewGrid:
    def __init__(self, title="Unified Preview", history_len=600, target_fps=30):
        self.title = title
        self.history_len = int(max(120, history_len))
        self.target_dt = 1.0 / float(max(5, target_fps))
        import threading
        self.lock = threading.Lock()
        self._run_flag = threading.Event()
        self._is_open = threading.Event()
        self._thread = None
        self.cell_h, self.cell_w = DEFAULT_CELL_H, DEFAULT_CELL_W
        self.hand_frame = None
        self.emo_frame = None
        self.r0_cum_hist = deque(maxlen=self.history_len)
        self.r0_last_avg_speed = 0.0
        self.va_hist = deque(maxlen=self.history_len)

    # ---------- external ----------
    def start(self):
        if self._thread: return
        self._run_flag.set(); self._is_open.set()
        import threading
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def reopen_window(self): self._is_open.set()
    def close_window(self):
        self._is_open.clear()
        try: cv2.destroyWindow(self.title)
        except: pass
    def stop(self):
        self._run_flag.clear(); self._is_open.clear()
        try: cv2.destroyWindow(self.title)
        except: pass

    def update_hand_frame(self, img): 
        with self.lock: self.hand_frame = img.copy(); self._maybe_set_cell_size(img)
    def update_emo_frame(self, img):
        with self.lock: self.emo_frame = img.copy(); self._maybe_set_cell_size(img)
    def push_r0_cumulative(self, cum, avg):
        with self.lock: self.r0_cum_hist.append(float(cum)); self.r0_last_avg_speed = float(avg)
    def push_valence_arousal(self, val, aro):
        with self.lock: self.va_hist.append((float(val), float(aro)))

    def _maybe_set_cell_size(self, img):
        h, w = img.shape[:2]
        maxw = 640
        scale = min(1.0, maxw / float(w)) if w>0 else 1.0
        self.cell_w, self.cell_h = int(w*scale), int(h*scale)

    # ---------- badges ----------
    def _badge(self, img, text, corner="tl", pad=8, opacity=0.4):
        """Rounded badge, larger spacing to avoid overlay collision"""
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bw, bh = tw + 18, th + 14
        H, W = img.shape[:2]
        if corner=="tl": x0,y0=pad, pad
        elif corner=="tr": x0,y0=W-bw-pad, pad
        elif corner=="bl": x0,y0=pad,H-bh-pad
        else: x0,y0=W-bw-pad,H-bh-pad
        overlay = img.copy()
        cv2.rectangle(overlay,(x0,y0),(x0+bw,y0+bh),(0,0,0),-1)
        cv2.addWeighted(overlay, opacity, img, 1-opacity, 0, img)
        cv2.rectangle(img,(x0,y0),(x0+bw,y0+bh),(180,180,180),1)
        cv2.putText(img,text,(x0+9,y0+bh-5),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
        return img

    # ---------- cumulative plot ----------
    def _render_plot_line(self, values, label, avg_speed=0.0, center_tip=True):
        W,H=self.cell_w,self.cell_h
        canvas=np.zeros((H,W,3),np.uint8)
        x0,y0,x1,y1=46,28,W-26,H-46
        width,height=max(1,x1-x0),max(1,y1-y0)
        cv2.rectangle(canvas,(x0,y0),(x1,y1),(80,80,80),1)
        self._badge(canvas,label,"tl")
        if not values: return canvas
        v_tip=float(values[-1])
        ymin=min(values); ymax=max(values)
        half=max(abs(v_tip-ymin),abs(v_tip-ymax))*1.1
        half=max(half,1e-3); ymin=v_tip-half; ymax=v_tip+half
        def vy(v): return int(y1-(v-ymin)/(ymax-ymin)*(y1-y0))
        n=len(values); cx=x0+width//2
        pts=[]
        for i,v in enumerate(values):
            frac=i/(n-1) if n>1 else 0
            x=int(x0+frac*(cx-x0)) if center_tip else int(x0+frac*(x1-x0))
            pts.append((x,vy(v)))
        if center_tip: cv2.line(canvas,(cx,y0),(cx,y1),(64,64,64),1)
        for i in range(1,len(pts)): cv2.line(canvas,pts[i-1],pts[i],(0,255,255),2)
        tip=pts[-1]
        cv2.circle(canvas,tip,3,(255,255,255),-1)
        # Tip text restored
        txt=f"{v_tip:.1f} mm | avg:{avg_speed:.2f} mm/s"
        cv2.putText(canvas,txt,(tip[0]+10,tip[1]-6),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
        return canvas

    # ---------- valence/arousal ----------
    def _render_va(self, tuples_va):
        W,H=self.cell_w,self.cell_h
        canvas=np.zeros((H,W,3),np.uint8)
        x0,y0,x1,y1=46,28,W-26,H-46
        cv2.rectangle(canvas,(x0,y0),(x1,y1),(80,80,80),1)
        self._badge(canvas,"Valence (Green)  Arousal (Orange)","tr")
        if not tuples_va: return canvas
        def vy_val(v): return int(y1-(v+1)/2*(y1-y0))
        def vy_aro(a): return int(y1-(a)*(y1-y0))
        prev_v=prev_a=None
        n=len(tuples_va)
        for i,(v,a) in enumerate(tuples_va):
            x=x0+int(i*(x1-x0-1)/(n-1))
            yv,ya=vy_val(v),vy_aro(a)
            if prev_v: cv2.line(canvas,prev_v,(x,yv),(0,255,0),2)
            if prev_a: cv2.line(canvas,prev_a,(x,ya),(0,165,255),2)
            prev_v,prev_a=(x,yv),(x,ya)
        return canvas

    # ---------- compose ----------
    def _compose_grid(self, hand, emo, mov_plot, va_plot):
        def fit(img):
            if img is None: return np.zeros((self.cell_h,self.cell_w,3),np.uint8)
            return cv2.resize(img,(self.cell_w,self.cell_h))
        tl, tr, bl, br = map(fit,(hand,emo,mov_plot,va_plot))
        # badges positioned to avoid overlap
        self._badge(tl,"Movement Camera","tr")   # moved to top-right now
        self._badge(tr,"Emotion Camera","tr")
        self._badge(bl,"Cumulative R_0","bl")
        self._badge(br,"Affect Traces","br")
        grid=np.zeros((self.cell_h*2,self.cell_w*2,3),np.uint8)
        grid[0:self.cell_h,0:self.cell_w]=tl
        grid[0:self.cell_h,self.cell_w:]=tr
        grid[self.cell_h:,0:self.cell_w]=bl
        grid[self.cell_h:,self.cell_w:]=br
        return grid

    # ---------- loop ----------
    def _loop(self):
        from control_flags import stop_event
        while self._run_flag.is_set():
            if self._is_open.is_set():
                with self.lock:
                    hand=self.hand_frame.copy() if self.hand_frame is not None else None
                    emo=self.emo_frame.copy() if self.emo_frame is not None else None
                    r0=list(self.r0_cum_hist); avg=self.r0_last_avg_speed
                    va=list(self.va_hist)
                mov=self._render_plot_line(r0,"R_0 cumulative (mm)",avg,center_tip=True)
                va_plot=self._render_va(va)
                grid=self._compose_grid(hand,emo,mov,va_plot)
                cv2.imshow(self.title,grid)
                key=cv2.waitKey(1)&0xFF
                if key==ord('q'): self.close_window()
                elif key==27:
                    print("[⛔] ESC (window) → stopping."); stop_event.set()
            time.sleep(self.target_dt)
