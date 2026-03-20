# -*- coding: utf-8 -*-
"""
【GitHub Actions自動化版】難易度アップ版：4歳児向け〈合同図形さがし〉プリント
"""

import os, random, math, pathlib, datetime, shutil
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- 自動化用ライブラリ ---
import smtplib
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials # ★変更点
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ---------------------------------------------------------
# 各種設定（ユーザー変更部分）
# ---------------------------------------------------------
DRIVE_PARENT_FOLDER_ID = "1-LxD713KndMDQWPPMnot3OS-MxQ3obrP" # 保存先フォルダID
NOTIFY_EMAIL_TO = "sk.green.abcj@gmail.com"                  # 通知先アドレス

BASE_SCALE = 75

# ──────────────────────────────────
# 1. 図形ファクトリ（変更なし）
# ──────────────────────────────────
def rect(w=2.2, h=1.4): return np.array([[-w/2,-h/2],[w/2,-h/2],[w/2,h/2],[-w/2,h/2]])
def square(a=1.6): return np.array([[-a/2,-a/2],[a/2,-a/2],[a/2,a/2],[-a/2,a/2]])
def right_tri(a=2.0, b=1.6): return np.array([[-a/3,-b/3],[2*a/3,-b/3],[-a/3,2*b/3]])
def iso_tri(b=2.2, h=1.8): return np.array([[-b/2,-h/3],[b/2,-h/3],[0,2*h/3]])
def trapezoid(top=1.2, bottom=2.4, h=1.4): return np.array([[-bottom/2,-h/2],[bottom/2,-h/2],[top/2,h/2],[-top/2,h/2]])
def right_trap(bottom=2.2, top=1.2, h=1.4): return np.array([[-bottom/2,-h/2],[bottom/2,-h/2],[top-bottom/2,h/2],[-bottom/2,h/2]])
def parallelogram(w=2.0, h=1.4, s=0.8): return np.array([[-w/2-s/2,-h/2],[w/2-s/2,-h/2],[w/2+s/2,h/2],[-w/2+s/2,h/2]])
def rhombus(w=2.2, h=1.6): return np.array([[0,-h/2],[w/2,0],[0,h/2],[-w/2,0]])
def ellipse(rx=1.6, ry=1.0, n=80):
    ang=np.linspace(0,2*math.pi,n,endpoint=False)
    return np.column_stack([rx*np.cos(ang),ry*np.sin(ang)])
def regular_ngon(n=6, r=1.4):
    ang=np.linspace(0,2*math.pi,n,endpoint=False)+math.pi/n
    return np.column_stack([r*np.cos(ang),r*np.sin(ang)])

SHAPES = {
    "rectangle":rect, "square":square, "right_tri":right_tri, "iso_tri":iso_tri,
    "trapezoid":trapezoid, "right_trap":right_trap,
    "parallelogram":parallelogram, "rhombus":rhombus, "ellipse":ellipse
}
ADVANCED_TYPES = {"trapezoid","right_trap","parallelogram","rhombus","ellipse","ngon"}

# ──────────────────────────────────
# 2. ユーティリティ（変更なし）
# ──────────────────────────────────
def rotate(poly:np.ndarray, deg:float)->np.ndarray:
    rad=math.radians(deg)
    R=np.array([[math.cos(rad),-math.sin(rad)],[math.sin(rad), math.cos(rad)]])
    return (R@poly.T).T

def get_all_distances(poly:np.ndarray)->List[float]:
    n = len(poly)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            dists.append(np.linalg.norm(poly[i] - poly[j]))
    return sorted(dists)

def poly_area(p:np.ndarray)->float:
    return 0.5 * np.abs(np.dot(p[:,0], np.roll(p[:,1], 1)) - np.dot(p[:,1], np.roll(p[:,0], 1)))

def is_congruent(p1:np.ndarray, p2:np.ndarray, tol=3.0)->bool:
    if len(p1) != len(p2): return False
    if abs(poly_area(p1) - poly_area(p2)) > tol * 20: return False
    if len(p1) <= 12:
        d1, d2 = get_all_distances(p1), get_all_distances(p2)
        if not all(abs(a - b) < tol for a, b in zip(d1, d2)):
            return False
    return True

def visible_rotation(poly:np.ndarray, rnd:random.Random)->float:
    w0, h0 = np.ptp(poly, axis=0)
    for _ in range(10):
        ang = rnd.choice([45, 90, 135, 180, 225, 270, 315])
        w, h = np.ptp(rotate(poly, ang), axis=0)
        if abs(w-w0)/w0 > 0.05 or abs(h-h0)/h0 > 0.05:
            return ang
    return rnd.choice([90, 180, 270])

# ──────────────────────────────────
# 3. １ページ生成（変更なし）
# ──────────────────────────────────
def generate_one(page:int, save_dir:pathlib.Path):
    rnd = random.Random()
    shape_keys = list(SHAPES.keys()) + ["ngon"]

    while True:
        k1, k2 = rnd.sample(shape_keys, 2)
        def make_shape(k):
            if k == "ngon":
                n = rnd.choice([5,6,7,8])
                return (k, n, regular_ngon(n) * BASE_SCALE)
            elif k == "ellipse":
                return (k, None, ellipse(rx=rnd.uniform(1.4,1.8), ry=rnd.uniform(0.9,1.2)) * BASE_SCALE)
            return (k, None, SHAPES[k]() * BASE_SCALE)
        s1, s2 = make_shape(k1), make_shape(k2)
        if (k1 in ADVANCED_TYPES or k2 in ADVANCED_TYPES) and not is_congruent(s1[2], s2[2]):
            specified = [s1, s2]
            break

    correct = [rotate(p, visible_rotation(p, rnd)) for _, _, p in specified]

    dummies_raw = []
    for s_type, aux, base in specified:
        base_max_dim = max(np.ptp(base, axis=0))
        if s_type == "ngon":
            diff = rnd.choice([-1, 1, 2])
            ns = max(3, aux + diff) 
            d_ngon = regular_ngon(ns) * BASE_SCALE
            dummies_raw.append((d_ngon, base_max_dim))
            r1, r2 = rnd.choice([(2.0, 0.5), (0.5, 2.0)])
            dummies_raw.append((base * np.array([r1, r2]), base_max_dim))
        else:
            r1, r2 = rnd.choice([(2.0, 0.5), (0.5, 2.0)])
            dummies_raw.append((base * np.array([r1, r2]), base_max_dim))
            if s_type in {"parallelogram", "rhombus", "rectangle", "trapezoid", "right_trap", "square"}:
                factor = rnd.uniform(0.8, 1.2) * rnd.choice([1, -1])
                shear = base.copy()
                shear[:, 0] += shear[:, 1] * factor
                shear[:, 1] *= rnd.choice([0.4, 0.6]) 
                dummies_raw.append((shear, base_max_dim))
            elif s_type == "ellipse":
                rx_ry = rnd.choice([(2.5, 0.4), (0.4, 2.5), (1.5, 1.5)])
                d = ellipse(rx=rx_ry[0], ry=rx_ry[1]) * BASE_SCALE
                dummies_raw.append((d, base_max_dim))

    dummies = []
    for d, b_max in dummies_raw:
        d_max = max(np.ptp(d, axis=0))
        if d_max > 0:
            d = d * (b_max / d_max) 
            dummies.append(d)

    valid_dummies = []
    for d in dummies:
        if len(valid_dummies) >= 4: break
        if all(not is_congruent(d, c) for c in correct) and \
           all(not is_congruent(d, v) for v in valid_dummies) and \
           all(not is_congruent(d, sp[2]) for sp in specified):
            valid_dummies.append(d)

    pool_keys = [k for k in shape_keys if k not in {k1, k2}]
    attempt = 0
    while len(valid_dummies) < 4 and attempt < 50:
        attempt += 1
        k = rnd.choice(pool_keys)
        if k == "ngon":
            p = regular_ngon(rnd.choice([5,6,7,8])) * BASE_SCALE * rnd.uniform(0.9, 1.1)
        elif k == "ellipse":
            p = ellipse(rx=rnd.uniform(1.3, 1.8), ry=rnd.uniform(0.9, 1.3)) * BASE_SCALE
        else:
            p = SHAPES[k]() * BASE_SCALE * rnd.uniform(0.9, 1.1)
        if all(not is_congruent(p, c) for c in correct) and \
           all(not is_congruent(p, v) for v in valid_dummies) and \
           all(not is_congruent(p, sp[2]) for sp in specified):
            valid_dummies.append(p)

    valid_dummies = valid_dummies[:4]
    all_polys = [sp[2] for sp in specified] + correct + valid_dummies
    max_dim = max(max(np.ptp(p, axis=0)) for p in all_polys)
    SAFE_SIZE = 230.0
    global_scale = SAFE_SIZE / max_dim if max_dim > 0 else 1.0

    W, H = 1280, 720
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect('equal'); ax.axis('off')
    try: import japanize_matplotlib
    except: pass

    ax.text(W/2, H-50, '「ひだり」のずけいとおなじものを「みぎ」からえらぼう',
            ha='center', va='center', fontsize=32, fontweight='bold')
    
    split = W * 0.33
    ax.plot([split, split], [H-110, 0], color='black', linewidth=4)

    def draw(poly, center):
        p = poly * global_scale
        p = p - p.mean(axis=0) + np.array(center)
        ax.add_patch(mpatches.Polygon(p, fill=False, edgecolor='black', linewidth=4, joinstyle='round'))

    Y_TOP = H * 0.65    
    Y_BOTTOM = H * 0.28 

    for poly, c in zip([sp[2] for sp in specified], [(split/2, Y_TOP), (split/2, Y_BOTTOM)]):
        draw(poly, c)

    choices = correct + valid_dummies
    rnd.shuffle(choices)
    centres = [(split + (W-split)*(c+0.5)/3, Y_TOP if r==0 else Y_BOTTOM)
               for r in range(2) for c in range(3)]
    for p, c in zip(choices, centres):
        draw(p, c)

    fname = save_dir / f"worksheet_{page:02d}.png"
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)

# ──────────────────────────────────
# 4. 追加機能：Google Drive API アップロード（★OAuthリフレッシュトークン対応）
# ──────────────────────────────────
def get_drive_service():
    """環境変数からOAuth情報を読み込み、Drive APIサービスを返す"""
    client_id = os.environ.get('GCP_CLIENT_ID')
    client_secret = os.environ.get('GCP_CLIENT_SECRET')
    refresh_token = os.environ.get('GCP_REFRESH_TOKEN')
    
    if not all([client_id, client_secret, refresh_token]):
        raise ValueError("環境変数(GCP_CLIENT_ID, GCP_CLIENT_SECRET, GCP_REFRESH_TOKEN)が設定されていません。")
    
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri='https://oauth2.googleapis.com/token',
        client_id=client_id,
        client_secret=client_secret
    )
    return build('drive', 'v3', credentials=creds)

def upload_to_gdrive(local_dir: pathlib.Path, folder_id: str):
    print("\nGoogle Driveへ接続しています...")
    drive_service = get_drive_service()

    today_str = datetime.datetime.now().strftime("%Y%m%d")
    
    folder_metadata = {
        'name': today_str,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [folder_id]
    }
    folder = drive_service.files().create(body=folder_metadata, fields='id').execute()
    new_folder_id = folder.get('id')
    print(f"Driveにフォルダ [{today_str}] を作成しました。")

    img_files = list(local_dir.glob("*.png"))
    for file_path in img_files:
        file_metadata = {'name': file_path.name, 'parents': [new_folder_id]}
        media = MediaFileUpload(str(file_path), mimetype='image/png', resumable=True)
        drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"Driveへアップロード完了: {file_path.name}")
        
    return today_str

# ──────────────────────────────────
# 5. 追加機能：メール通知
# ──────────────────────────────────
def send_completion_email(folder_name: str, file_count: int):
    sender_email = os.environ.get('GMAIL_ADDRESS')
    app_password = os.environ.get('GMAIL_APP_PASS')
    
    if not sender_email or not app_password:
        print("\n⚠️ 環境変数(GMAIL_ADDRESS, GMAIL_APP_PASS)が設定されていないため、メール通知をスキップします。")
        return

    subject = f"【自動生成完了】図形プリント ({folder_name})"
    body = f"""
お疲れ様です。
今週分の図形プリント（{file_count}枚）の生成が完了しました。

Google Driveのフォルダ「{folder_name}」に保存されています。
ご確認ください。
"""
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = NOTIFY_EMAIL_TO

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()
        print("\n✉️ 完了通知メールを送信しました！")
    except Exception as e:
        print(f"\n❌ メール送信に失敗しました: {e}")

# ──────────────────────────────────
# 6. メイン（一括処理）
# ──────────────────────────────────
def main(n_pages:int=3):
    out_dir = pathlib.Path.cwd() / "temp_worksheets"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)
    
    print(f"{n_pages}枚のプリントを作成中...")
    for i in range(1, n_pages+1):
        generate_one(i, out_dir)
        print(f"作成完了: worksheet_{i:02d}.png")

    created_folder = upload_to_gdrive(out_dir, DRIVE_PARENT_FOLDER_ID)
    send_completion_email(created_folder, n_pages)

    shutil.rmtree(out_dir)
    print("\n🗑️ 一時ディレクトリを削除しました。")
    print("✨ すべての自動化プロセスが完了しました！")

if __name__=="__main__":
    main(10) # デフォルトで3枚生成
