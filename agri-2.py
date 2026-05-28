import os
import streamlit as st
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

# ─── CONFIGURATION DE LA PAGE ─────────────────────────────────────────
st.set_page_config(
    page_title="Agri'Mali",
    page_icon="🌿",
    layout="centered"
)

st.markdown("""
    <style>
    .main { background-color: #f7f9f7; }

    div.stButton > button:first-child {
        background-color: #1e824c;
        color: white;
        border-radius: 25px;
        padding: 12px 24px;
        font-weight: bold;
        border: none;
        width: 100%;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        font-size: 16px;
    }
    div.stButton > button:first-child:hover {
        background-color: #155d35;
        transform: translateY(-1px);
    }

    .status-card-healthy {
        background-color: #2e7d32;
        color: white;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(46, 125, 50, 0.3);
        margin-bottom: 15px;
    }
    .status-card-sick {
        background-color: #d32f2f;
        color: white;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(211, 47, 47, 0.3);
        margin-bottom: 15px;
    }
    .hint-text {
        background-color: #fef1b8;
        color: #5c3a00;
        padding: 12px;
        border-radius: 8px;
        font-weight: 500;
        margin-bottom: 15px;
    }
    .confidence-bar-bg {
        background-color: #e0e0e0;
        border-radius: 8px;
        height: 12px;
        margin-top: 4px;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)


# ─── CHARGEMENT DU MODÈLE ET DES LABELS ───────────────────────────────
@st.cache_resource
def load_local_model():
    model_path = "agrimali_best.pth"
    labels_path = "labels.txt"

    # Vérification des fichiers nécessaires
    if not os.path.exists(model_path):
        st.error(f"❌ Fichier modèle '{model_path}' introuvable. Placez-le dans le même dossier que app.py.")
        return None, None

    if not os.path.exists(labels_path):
        st.error(f"❌ Fichier '{labels_path}' introuvable. Placez-le dans le même dossier que app.py.")
        return None, None

    with open(labels_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    num_classes = len(classes)

    if num_classes == 0:
        st.error("❌ Le fichier labels.txt est vide.")
        return None, None

    # Construction du modèle (weights=None remplace pretrained=False, déprécié)
    model = models.mobilenet_v2(weights=None)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(1280, 512),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(512),
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(512, num_classes)
    )

    # Chargement du checkpoint — compatible dict {"model": ...} ET state_dict direct
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        # weights_only non supporté sur les vieilles versions de PyTorch
        checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Le fichier .pth est directement un state_dict
        state_dict = checkpoint

    # Nettoyage des clés (supprime le préfixe "backbone." si présent)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[9:] if k.startswith("backbone.") else k
        cleaned_state_dict[new_key] = v

    try:
        model.load_state_dict(cleaned_state_dict)
    except RuntimeError as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None, None

    model.eval()
    return model, classes


# ─── FONCTION D'INFÉRENCE ─────────────────────────────────────────────
def predict(image: Image.Image, model, classes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Top-3 prédictions
    top3_probs, top3_idx = torch.topk(probabilities, min(3, len(classes)))
    results = []
    for prob, idx in zip(top3_probs.tolist(), top3_idx.tolist()):
        raw_label = classes[idx]
        clean = raw_label.split("___")[-1].replace("_", " ").title() if "___" in raw_label else raw_label.replace("_", " ").title()
        is_healthy = "healthy" in raw_label.lower()
        results.append({
            "label": clean,
            "raw": raw_label,
            "confidence": prob * 100,
            "is_healthy": is_healthy
        })
    return results


# ─── INTERFACE UTILISATEUR ────────────────────────────────────────────
st.markdown("<h1 style='text-align:center; color:#2e7d32; font-family:sans-serif;'>🌿 Agri'Mali</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666; margin-top:-15px;'>Votre assistant agricole intelligent</p>", unsafe_allow_html=True)
st.markdown("---")

model, classes = load_local_model()

if model is None:
    st.stop()

st.markdown('<div class="hint-text">📸 Prenez une photo claire de la feuille de votre plante, ou importez une image depuis votre galerie.</div>', unsafe_allow_html=True)

# ─── CORRECTION DU BUG DES TABS ───────────────────────────────────────
# Chaque tab utilise SA PROPRE variable pour éviter l'écrasement mutuel.
# On utilise session_state pour conserver l'image active entre les reruns.

tab_camera, tab_gallery = st.tabs(["📸 Prendre une photo", "🖼️ Importer une image"])

with tab_camera:
    camera_file = st.camera_input("Appareil photo")
    if camera_file is not None:
        st.session_state["active_image"] = camera_file
        st.session_state["image_source"] = "camera"

with tab_gallery:
    gallery_file = st.file_uploader("Importer depuis la galerie", type=["jpg", "jpeg", "png"])
    if gallery_file is not None:
        st.session_state["active_image"] = gallery_file
        st.session_state["image_source"] = "gallery"

# ─── AFFICHAGE ET ANALYSE ─────────────────────────────────────────────
active_file = st.session_state.get("active_image", None)
image_source = st.session_state.get("image_source", None)

if active_file is not None:
    try:
        image = Image.open(active_file).convert("RGB")
    except Exception as e:
        st.error(f"❌ Impossible d'ouvrir l'image : {e}")
        st.stop()

    # N'afficher l'image que si elle vient de la galerie
    # (la camera_input l'affiche déjà nativement dans son tab)
    if image_source == "gallery":
        st.image(image, caption="Image sélectionnée", use_container_width=True)

    st.write("")
    if st.button("🔍 ANALYSER LA FEUILLE"):
        with st.spinner("🔬 Analyse en cours..."):
            results = predict(image, model, classes)

        top = results[0]
        st.write("### Résultat")

        if top["is_healthy"]:
            st.markdown(f"""
                <div class="status-card-healthy">
                    <div style='font-size:50px;'>😊</div>
                    <h2 style='margin:5px 0 0 0;'>PLANTE SAINE</h2>
                    <p style='margin:5px 0 0 0; opacity:0.9;'>Aucune maladie détectée</p>
                    <h3 style='margin:10px 0 0 0;'>Confiance : {top['confidence']:.0f}%</h3>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
            st.info("🌿 Bonne nouvelle ! Continue à bien entretenir ta plante. Arrose régulièrement et garde ton champ propre.")

        else:
            st.markdown(f"""
                <div class="status-card-sick">
                    <div style='font-size:50px;'>🙁</div>
                    <h2 style='margin:5px 0 0 0;'>PLANTE MALADE</h2>
                    <p style='margin:5px 0 0 0; opacity:0.9;'>{top['label']}</p>
                    <h3 style='margin:10px 0 0 0;'>Confiance : {top['confidence']:.0f}%</h3>
                </div>
            """, unsafe_allow_html=True)
            st.error(f"**Maladie détectée : {top['label']}**\n\nCette maladie peut réduire la production si elle n'est pas traitée à temps.")
            st.warning(
                "**Que faire ?**\n\n"
                "• Élimine les feuilles très touchées\n"
                "• Utilise un fongicide recommandé\n"
                "• Évite l'excès d'eau\n"
                "• Surveille régulièrement ta plante\n"
                "• Consulte un agent agricole si les symptômes persistent"
            )

        # ─── TOP 3 DES PRÉDICTIONS ────────────────────────────────────
        if len(results) > 1:
            st.write("### 📊 Top 3 des prédictions")
            for i, r in enumerate(results):
                emoji = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")
                color = "#2e7d32" if r["is_healthy"] else "#d32f2f"
                bar_width = int(r["confidence"])
                st.markdown(
                    f"**{emoji} {r['label']}** — {r['confidence']:.1f}%"
                )
                st.markdown(
                    f'<div class="confidence-bar-bg"><div style="background:{color};height:12px;border-radius:8px;width:{bar_width}%"></div></div>',
                    unsafe_allow_html=True
                )

else:
    st.markdown(
        "<p style='text-align:center; color:#aaa; margin-top:30px;'>📷 Aucune image sélectionnée</p>",
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown("<p style='text-align:center; color:#aaa; font-size:12px;'>Agri'Mali — Développé pour les agriculteurs maliens 🇲🇱</p>", unsafe_allow_html=True)
