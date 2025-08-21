# rag_chatbot_bank_embed_gemini.py
# ----------------------------------------------------
# Prototype: RAG Chatbot + ML Recommendation (Banking)
# - Multi-label Logistic Regression cho gợi ý sản phẩm
# - RAG bằng Google Gemini embeddings + cosine similarity
# ----------------------------------------------------

import os
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai


# =========================
# 1) Setup Gemini
# =========================
genai.configure(api_key="xxx")

def get_embedding(text: str, model: str = "models/text-embedding-004") -> np.ndarray:
    res = genai.embed_content(
        model=model,
        content=text,
        task_type="retrieval_document"  # hoặc "classification"
    )
    return np.array(res["embedding"])


# =========================
# 2) Dummy DATA
# =========================

CUSTOMERS = pd.DataFrame({
    "cus_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "segment": ["Mass", "Mass", "Affluent", "Mass", "Affluent", "Mass", "Affluent", "Mass"],
    "age": [28, 35, 42, 31, 55, 26, 48, 33],
    "income_band": ["20-30k", "30-40k", "60-80k", "30-40k", "80-100k", "20-30k", "100-120k", "40-50k"],
    "products": [
        ["checking_account", "debit_card"],                         # khách hàng cơ bản
        ["checking_account"],                                       # cơ bản
        ["checking_account", "credit_card"],                        # đã có cross-sell 1
        ["checking_account", "debit_card", "credit_card"],          # cross-sell
        ["checking_account", "savings"],                            # tiết kiệm
        ["checking_account", "debit_card", "insurance"],            # bảo hiểm
        ["checking_account", "credit_card", "investment_fund"],     # đầu tư
        ["checking_account", "savings", "personal_loan"]            # vay cá nhân
    ]
})

ALL_PRODUCTS = [
    "checking_account", "debit_card", "credit_card", "savings",
    "term_deposit", "personal_loan", "insurance", "investment_fund", "mobile_banking"
]

PRODUCT_KB = [
    {"product": "checking_account", "text": "Tài khoản thanh toán dùng để nhận lương, chuyển khoản, thanh toán hoá đơn. Phí duy trì thấp, đi kèm Mobile Banking."},
    {"product": "debit_card", "text": "Thẻ ghi nợ nội địa/quốc tế, chi tiêu trong số dư tài khoản, rút tiền ATM, miễn phí rút tại ATM nội bộ."},
    {"product": "credit_card", "text": "Thẻ tín dụng cho phép chi tiêu trước trả sau, miễn lãi tối đa 45 ngày, có hoàn tiền/dặm bay tùy hạng thẻ."},
    {"product": "savings", "text": "Tiết kiệm online kỳ hạn linh hoạt 1-12 tháng, lãi suất bậc thang theo số dư và kỳ hạn."},
    {"product": "term_deposit", "text": "Tiền gửi có kỳ hạn lãi suất cao hơn tiết kiệm thường, tất toán trước hạn hưởng lãi không kỳ hạn."},
    {"product": "personal_loan", "text": "Vay tín chấp dựa vào thu nhập, hạn mức 3-12 lần lương, thời hạn đến 60 tháng, phê duyệt nhanh."},
    {"product": "insurance", "text": "Bảo hiểm nhân thọ/sức khỏe, bảo vệ tài chính trước rủi ro, linh hoạt quyền lợi và phí."},
    {"product": "investment_fund", "text": "Quỹ mở dành cho khách ít thời gian, đa dạng rủi ro-lợi suất (trái phiếu/cân bằng/cổ phiếu)."},
    {"product": "mobile_banking", "text": "Ứng dụng ngân hàng di động: chuyển khoản 24/7, QR, tiết kiệm online, thanh toán hóa đơn."}
]
KB_DF = pd.DataFrame(PRODUCT_KB)


# =========================
# 3) Dummy recommender (rule-based)
# =========================
# Đã bỏ hàm recommend_next_products, thay bằng dummy data kết quả gợi ý
CUSTOMER_RECOMMEND = {
    1: [("credit_card", 0.9), ("savings", 0.8)],
    2: [("debit_card", 0.85), ("savings", 0.75)],
    3: [("investment_fund", 0.88), ("insurance", 0.77)],
    4: [("term_deposit", 0.82), ("personal_loan", 0.7)],
    5: [("credit_card", 0.8), ("insurance", 0.65)],
    6: [("investment_fund", 0.9), ("mobile_banking", 0.8)],
    7: [("savings", 0.85), ("term_deposit", 0.75)],
    8: [("insurance", 0.88), ("credit_card", 0.78)],
}


# =========================
# 4) RAG: Gemini embedding retriever
# =========================
kb_embeddings = np.vstack([get_embedding(txt) for txt in KB_DF["text"].tolist()])

def rag_retrieve(query: str, top_k: int = 2) -> List[Dict]:
    q_emb = get_embedding(query)
    sims = cosine_similarity([q_emb], kb_embeddings).ravel()
    idx = sims.argsort()[::-1][:top_k]
    return [
        {"product": KB_DF.iloc[i]["product"], "passage": KB_DF.iloc[i]["text"], "score": float(sims[i])}
        for i in idx
    ]


# =========================
# 5) Helpers
# =========================
def get_customer(cus_id: int) -> Dict:
    row = CUSTOMERS[CUSTOMERS["cus_id"] == cus_id]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


# =========================
# 6) Simple chatbot
# =========================
CUS_ID_PAT = re.compile(r"(?:kh|khách\s*hàng|customer)\s*[:#\s]*?(\d+)", re.I)

def extract_cus_id(text: str) -> int:
    m = CUS_ID_PAT.search(text)
    if m: return int(m.group(1))
    nums = re.findall(r"\d+", text)
    return int(nums[0]) if nums else -1

def chat(user_input: str) -> str:
    cus_id = extract_cus_id(user_input)
    if cus_id == -1:
        return "Bạn vui lòng nhập mã khách hàng (vd: KH 3)."
    cust = get_customer(cus_id)
    if not cust:
        return f"Không tìm thấy khách hàng {cus_id}."

    owned = cust["products"]
    resp = [f"KH {cus_id} đang có: {', '.join(owned)}."]
    recos = CUSTOMER_RECOMMEND.get(cus_id, [])
    if recos:
        reco_str = ", ".join([f"{p} (p≈{s:.2f})" for p, s in recos])
        resp.append(f"Gợi ý bán thêm: {reco_str}.")
        top_prod = recos[0][0]
        rag_hits = rag_retrieve(top_prod.replace("_", " "))
        if rag_hits:
            explain = "; ".join([h['passage'] for h in rag_hits])
            resp.append(f"Lý do: {explain}")
    else:
        resp.append("Chưa có gợi ý thêm.")

    return "\n".join(resp)


# =========================
# 7) Demo CLI
# =========================
if __name__ == "__main__":
    print(">>> Demo chatbot RAG + Gemini Embedding. Hỏi thử:\n- 'KH 3 đang có gì, nên bán gì thêm?'\n- 'Tư vấn cross sell cho khách hàng 2'\n")
    while True:
        q = input("Bạn: ").strip()
        if q.lower() in ["exit", "quit"]:
            break
        print("\nBot:", chat(q), "\n")
