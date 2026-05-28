import os
import streamlit as st
import torch
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms

# ─── CONFIGURATION DE LA PAGE STYLE WEB APP MOBILE ────────────────────
st.set_page_config(
    page_title="Agri'Mali",
    page_icon="🌿",
    layout="centered"
)

# Application de la charte graphique (Boutons arrondis, Vert Agri, Rouge Maladie)
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
    }
    .status-card-healthy {
        background-color: #2e7d32;
        color: white;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(46, 125, 50, 0.3);
    }
    .status-card-sick {
        background-color: #d32f2f;
        color: white;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(211, 47, 47, 0.3);
    }
    .hint-text {
        background-color: #fef1b8;
        color: #5c3a00;
        padding: 12px;
        border-radius: 8px;
        font-weight: 500;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ─── CHARGEMENT DU MODÈLE ET DES LABELS ───────────────────────────────
@st.cache_resource
def load_local_model():
    model_path = "agrimali_best.pth"
    labels_path = "labels.txt"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Fichier '{model_path}' introuvable dans ce dossier.")
        return None, None

    # Lecture des 38 labels
    with open(labels_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines()]
    num_classes = len(classes)

    # 1. Chargement du MobileNetV2 brut
    model = models.mobilenet_v2(pretrained=False)
    
    # 2. Reconstruction exacte du bloc classifier de ton entraînement AgriMaliNet
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(1280, 512),       # classifier.1 (Taille [512, 1280])
        torch.nn.ReLU(),                  # classifier.2
        torch.nn.BatchNorm1d(512),        # classifier.3 (running_mean, running_var...)
        torch.nn.Dropout(p=0.2),          # classifier.4
        torch.nn.Linear(512, num_classes) # classifier.5 (Sortie finale vers tes 38 classes)
    )
    
    # 3. Chargement et nettoyage des clés du dictionnaire de poids
    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint["model"]
    
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        # Enlever le préfixe 'backbone.' si présent
        new_key = k[9:] if k.startswith("backbone.") else k
        cleaned_state_dict[new_key] = v
        
    # Application des poids (Zéro erreur de dimension maintenant !)
    model.load_state_dict(cleaned_state_dict)
    model.eval()
    return model, classes

model, classes = load_local_model()

# ─── INTERFACE UTILISATEUR COMPATIBLE SMARTPHONE ─────────────────────
st.markdown("<h1 style='text-align: center; color: #2e7d32; font-family: sans-serif;'>Agri'Mali</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666; margin-top: -15px;'>Votre assistant agricole</p>", unsafe_allow_html=True)

# Message d'instruction imitant tes maquettes graphiques
st.markdown('<div class="hint-text">🔊 Si tu veux savoir l\'état de ta plante, prends une photo claire de sa feuille ou choisis un fichier.</div>', unsafe_allow_html=True)

# Sélection simplifiée de la source de l'image
source_mode = st.tabs(["📸 PRENDRE UNE PHOTO", "🖼️ CHOISIR UNE IMAGE"])

uploaded_file = None
with source_mode[0]:
    uploaded_file = st.camera_input("Caméra Appareil")
with source_mode[1]:
    uploaded_file = st.file_uploader("Importer depuis la galerie", type=["jpg", "jpeg", "png"])

# ─── TRAITEMENT ET INFÉRENCE IA ───────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Prétraitement de la feuille (Format attendu : 224x224 et Normalisation ImageNet)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    
    st.write("")
    if st.button("🔍 ANALYSER LA FEUILLE"):
        with st.spinner("Analyse en cours..."):
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, class_idx = torch.max(probabilities, dim=0)
            
            raw_label = classes[class_idx]
            confidence_score = confidence.item() * 100
            
            # Détection de l'état : sain (contient "healthy") ou atteint d'une pathologie
            is_healthy = "healthy" in raw_label.lower()
            
            # Nettoyage cosmétique du nom de la maladie pour l'utilisateur
            clean_name = raw_label.split("___")[-1].replace("_", " ") if "___" in raw_label else raw_label.replace("_", " ")
            
            st.write("### RÉSULTAT")
            
            if is_healthy:
                st.markdown(f"""
                    <div class="status-card-healthy">
                        <h1 style='margin:0; font-size: 50px;'>😊</h1>
                        <h2 style='margin:5px 0 0 0;'>SAINE</h2>
                        <p style='margin:5px 0 0 0; font-size:15px; opacity:0.9;'>La plante semble saine</p>
                        <h3 style='margin:10px 0 0 0;'>Confiance : {confidence_score:.0f}%</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.balloons()
                
                st.write("")
                st.subheader("💡 Conseils :")
                st.info("🌿 Bonne nouvelle ! Continue à bien entretenir ta plante pour garder sa santé. Arrose régulièrement et garde ton champ propre.")
            else:
                st.markdown(f"""
                    <div class="status-card-sick">
                        <h1 style='margin:0; font-size: 50px;'>🙁</h1>
                        <h2 style='margin:5px 0 0 0;'>MALADE</h2>
                        <p style='margin:5px 0 0 0; font-size:15px; opacity:0.9;'>La plante est malade</p>
                        <h3 style='margin:10px 0 0 0;'>Confiance : {confidence_score:.0f}%</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("")
                st.subheader("⚠️ Maladie détectée :")
                st.error(f"**{clean_name.title()}**\n\nCette maladie peut réduire la production si elle n'est pas traitée à temps.")
                
                st.subheader("📋 Que faire ?")
                st.warning("• Élimine les feuilles très touchées\n• Utilise un fongicide recommandé\n• Évite l'excès d'eau\n• Surveille régulièrement ta plante")