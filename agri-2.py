import os
import io
import base64
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from huggingface_hub import hf_hub_download
import timm
from gtts import gTTS

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
        background-color: #1e824c; color: white;
        border-radius: 25px; padding: 12px 24px;
        font-weight: bold; border: none; width: 100%;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1); font-size: 16px;
    }
    div.stButton > button:first-child:hover {
        background-color: #155d35; transform: translateY(-1px);
    }
    .status-card-healthy {
        background-color: #2e7d32; color: white;
        padding: 25px; border-radius: 16px; text-align: center;
        box-shadow: 0px 4px 12px rgba(46,125,50,0.3); margin-bottom: 15px;
    }
    .status-card-sick {
        background-color: #d32f2f; color: white;
        padding: 25px; border-radius: 16px; text-align: center;
        box-shadow: 0px 4px 12px rgba(211,47,47,0.3); margin-bottom: 15px;
    }
    .hint-text {
        background-color: #fef1b8; color: #5c3a00;
        padding: 12px; border-radius: 8px;
        font-weight: 500; margin-bottom: 15px;
    }
    .confidence-bar-bg {
        background-color: #e0e0e0; border-radius: 8px;
        height: 12px; margin-top: 4px; margin-bottom: 12px;
    }
    .bambara-box {
        background-color: #e8f5e9; color: #1b5e20;
        padding: 16px; border-radius: 12px;
        border-left: 5px solid #2e7d32;
        font-size: 15px; margin-top: 10px;
    }
    .audio-label {
        font-size: 13px; color: #555;
        margin-bottom: 4px; margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)


# ─── BASE DE DONNÉES MALADIES : FRANÇAIS + BAMBARA + CONSEILS ─────────
# Clés = suffixe PlantVillage (après "___"), en minuscules avec espaces
DISEASE_DB = {
    # ── TOMATE ──────────────────────────────────────────────────────────
    "tomato bacterial spot": {
        "fr": "Tache bactérienne de la tomate",
        "bam": "Tomati ni bana kɔrɔ  (tache noire)",
        "conseil_fr": (
            "• Retirez et brûlez les feuilles atteintes.\n"
            "• Appliquez un fongicide à base de cuivre.\n"
            "• Évitez d'arroser par le haut (arrosage au pied).\n"
            "• Espacez bien les plants pour favoriser la ventilation.\n"
            "• Consultez un agent agricole si ça s'aggrave."
        ),
        "conseil_bam": (
            "• Bɔ fɛnw minnu bana, i k'u jo.\n"
            "• Buli fɛn (cuivre) kɛ ni daji ye.\n"
            "• Jii bɔ sira fɛ, kana ji yɛlɛn kan.\n"
            "• Tomatiki to ka jan kelen kelen.\n"
            "• Nba nin ye gɛlɛya ye, wɛrɛ ɲinɛ."
        ),
    },
    "tomato early blight": {
        "fr": "Brûlure précoce de la tomate (Alternaria)",
        "bam": "Tomatiki ni bana (nɔgɔ kalan kɔrɔ)",
        "conseil_fr": (
            "• Éliminez les feuilles infectées dès que possible.\n"
            "• Appliquez un fongicide (mancozèbe ou chlorothalonil).\n"
            "• Faites une rotation des cultures chaque saison.\n"
            "• Évitez d'arroser le soir (humidité nocturne).\n"
            "• Enrichissez le sol en compost pour renforcer la plante."
        ),
        "conseil_bam": (
            "• Bɔ fɛnw minnu bana joona.\n"
            "• Buli fɛn (mancozèbe) kɛ ni daji ye.\n"
            "• Sannin kelen kelen, bɔ joo ye ka tɔ taabol la.\n"
            "• Kana ji bɔ su la.\n"
            "• Dugukolo ɲɛ ka kɛ ni fɔnitɔ ye."
        ),
    },
    "tomato late blight": {
        "fr": "Mildiou de la tomate",
        "bam": "Tomatiki ni ji bana (mildiou)",
        "conseil_fr": (
            "• Retirez immédiatement les parties touchées.\n"
            "• Appliquez un fongicide systémique (métalaxyl).\n"
            "• Évitez l'excès d'humidité et aérez le champ.\n"
            "• Ne composez pas les débris infectés.\n"
            "• Utilisez des variétés résistantes la prochaine saison."
        ),
        "conseil_bam": (
            "• Bɔ fɛnw minnu bana joona joona.\n"
            "• Buli fɛn (métalaxyl) kɛ ni daji ye.\n"
            "• Kana ji caman kɛ, ka jaa bɔ.\n"
            "• Kana ni fɛnw minnu bana ye ka kɛ fɔnitɔ.\n"
            "• Sɛgɛsɛgɛ tomatiki wɛrɛ minnu te bana."
        ),
    },
    "tomato leaf mold": {
        "fr": "Moisissure foliaire de la tomate",
        "bam": "Tomatiki ka furu bana",
        "conseil_fr": (
            "• Améliorez la ventilation du champ.\n"
            "• Appliquez un fongicide foliaire.\n"
            "• Réduisez l'arrosage et l'humidité ambiante.\n"
            "• Supprimez les feuilles très infectées."
        ),
        "conseil_bam": (
            "• Ka jaa kɛ ka diya yɔrɔ kɔnɔ.\n"
            "• Buli fɛn kɛ ka fɛn kan.\n"
            "• Jii nɔgɔya ka kɛ.\n"
            "• Bɔ fɛnw minnu bana kosɛbɛ."
        ),
    },
    "tomato septoria leaf spot": {
        "fr": "Septoriose de la tomate",
        "bam": "Tomatiki ni dɔtɔ bana (septoria)",
        "conseil_fr": (
            "• Retirez les feuilles du bas infectées.\n"
            "• Appliquez un fongicide à base de cuivre ou chlorothalonil.\n"
            "• Faites une rotation des cultures.\n"
            "• Évitez de toucher les plants mouillés."
        ),
        "conseil_bam": (
            "• Bɔ fɛnw minnu bana a kɔfɛ.\n"
            "• Buli fɛn (cuivre) kɛ ni daji ye.\n"
            "• Sannin taabol la.\n"
            "• Kana sen kɛ tomatiki jii tɔ."
        ),
    },
    "tomato spider mites two-spotted spider mite": {
        "fr": "Acariens (araignées rouges) sur tomate",
        "bam": "Tomatiki ni kurunsi fitinin bana",
        "conseil_fr": (
            "• Appliquez un acaricide ou de l'huile de neem.\n"
            "• Arrosez régulièrement (les acariens détestent l'humidité).\n"
            "• Évitez la poussière autour des plants.\n"
            "• Inspectez le dessous des feuilles régulièrement."
        ),
        "conseil_bam": (
            "• Buli fɛn (neem) kɛ ni daji ye.\n"
            "• Ji bɔ tuma o tuma.\n"
            "• Ka fura bɔ tomatiki kɔfɛ.\n"
            "• Sɛgɛsɛgɛ fɛnw kɔfɛ tuma o tuma."
        ),
    },
    "tomato target spot": {
        "fr": "Tache cible de la tomate",
        "bam": "Tomatiki ni gɛlɛn bana (cible)",
        "conseil_fr": (
            "• Retirez et détruisez les feuilles infectées.\n"
            "• Appliquez un fongicide (azoxystrobine).\n"
            "• Améliorez la circulation d'air dans le champ."
        ),
        "conseil_bam": (
            "• Bɔ fɛnw minnu bana k'u jo.\n"
            "• Buli fɛn kɛ ni daji ye.\n"
            "• Ka jaa kɛ ka diya yɔrɔ kɔnɔ."
        ),
    },
    "tomato tomato mosaic virus": {
        "fr": "Virus de la mosaïque de la tomate",
        "bam": "Tomatiki ni virus bana (mosaïque)",
        "conseil_fr": (
            "• Aucun traitement chimique efficace contre les virus.\n"
            "• Retirez et brûlez les plants infectés.\n"
            "• Contrôlez les insectes vecteurs (pucerons).\n"
            "• Utilisez des semences certifiées la prochaine fois.\n"
            "• Désinfectez vos outils agricoles."
        ),
        "conseil_bam": (
            "• Virus bana tɛ ko fɛ ka bɔ.\n"
            "• Bɔ tomatiki minnu bana k'u jo.\n"
            "• Buli fɛn kɛ kurunsi fitinin kan.\n"
            "• Sɛgɛsɛgɛ wari ka ɲɛ tɔɔrɔ kan.\n"
            "• Ka koloji kɛ sɛbɛnni kan."
        ),
    },
    "tomato tomato yellow leaf curl virus": {
        "fr": "Virus de l'enroulement jaune des feuilles de tomate",
        "bam": "Tomatiki ni jaala bana (feuille jaune)",
        "conseil_fr": (
            "• Retirez et détruisez les plants infectés.\n"
            "• Luttez contre les aleurodes (mouches blanches) vecteurs.\n"
            "• Utilisez des filets anti-insectes si possible.\n"
            "• Choisissez des variétés résistantes."
        ),
        "conseil_bam": (
            "• Bɔ tomatiki minnu bana k'u jo.\n"
            "• Buli fɛn kɛ fiɲɛfula fitinin kan.\n"
            "• Kɛ ni sɛbɛ ye ka bɔ kurunsi la.\n"
            "• Sɛgɛsɛgɛ tomatiki wɛrɛ."
        ),
    },
    "tomato healthy": {
        "fr": "Tomate saine",
        "bam": "Tomatiki ka kɛnɛya",
        "conseil_fr": "🌿 Votre plant de tomate est en bonne santé ! Continuez à l'entretenir régulièrement.",
        "conseil_bam": "🌿 I ka tomatiki ni kɛnɛya. Kɔlɔsi a ka ɲɛ.",
    },

    # ── MAÏS ─────────────────────────────────────────────────────────────
    "corn (maize) cercospora leaf spot gray leaf spot": {
        "fr": "Tache grise du maïs (Cercospora)",
        "bam": "Garikiki ni dɔtɔ bana (gris)",
        "conseil_fr": (
            "• Appliquez un fongicide foliaire adapté.\n"
            "• Faites une rotation avec le sorgho ou le mil.\n"
            "• Évitez les densités de plantation trop élevées.\n"
            "• Utilisez des variétés résistantes."
        ),
        "conseil_bam": (
            "• Buli fɛn kɛ garikiki kan.\n"
            "• Sannin ka kɛ ni sorow wala miji ye.\n"
            "• Kana garikiki to ka caman kosɛbɛ.\n"
            "• Sɛgɛsɛgɛ garikiki wɛrɛ."
        ),
    },
    "corn (maize) common rust": {
        "fr": "Rouille commune du maïs",
        "bam": "Garikiki ni sɔgɔ bana (rouille)",
        "conseil_fr": (
            "• Appliquez un fongicide (triazole ou strobilurine) dès les premiers symptômes.\n"
            "• Choisissez des semences de variétés résistantes.\n"
            "• Faites une rotation des cultures.\n"
            "• Évitez l'excès d'engrais azoté."
        ),
        "conseil_bam": (
            "• Buli fɛn kɛ joona tuma la.\n"
            "• Sɛgɛsɛgɛ garikiki wɛrɛ ka kɔrɔya.\n"
            "• Sannin taabol la.\n"
            "• Kana engrais caman kɛ."
        ),
    },
    "corn (maize) northern leaf blight": {
        "fr": "Brûlure du Nord du maïs (Helminthosporium)",
        "bam": "Garikiki ni bana kɔrɔ (nord)",
        "conseil_fr": (
            "• Retirez les résidus de récolte infectés.\n"
            "• Appliquez un fongicide systémique.\n"
            "• Utilisez des semences traitées et résistantes.\n"
            "• Pratiquez la rotation des cultures."
        ),
        "conseil_bam": (
            "• Bɔ fɛnw minnu bana ka sɔrɔ.\n"
            "• Buli fɛn kɛ ni daji ye.\n"
            "• Sɛgɛsɛgɛ wari ka ɲɛ.\n"
            "• Sannin taabol la."
        ),
    },
    "corn (maize) healthy": {
        "fr": "Maïs sain",
        "bam": "Garikiki ka kɛnɛya",
        "conseil_fr": "🌽 Votre plant de maïs est sain ! Continuez l'entretien.",
        "conseil_bam": "🌽 I ka garikiki ni kɛnɛya. Kɔlɔsi a ka ɲɛ.",
    },

    # ── POMME DE TERRE ───────────────────────────────────────────────────
    "potato early blight": {
        "fr": "Brûlure précoce de la pomme de terre",
        "bam": "Pomdetɛrɛ ni bana (kɔrɔ wɛrɛ)",
        "conseil_fr": (
            "• Retirez les feuilles infectées.\n"
            "• Appliquez un fongicide (mancozèbe).\n"
            "• Évitez les stress hydriques (arrosage régulier).\n"
            "• Faites une rotation des cultures."
        ),
        "conseil_bam": (
            "• Bɔ fɛnw minnu bana.\n"
            "• Buli fɛn kɛ ni daji ye.\n"
            "• Ji bɔ tuma o tuma.\n"
            "• Sannin taabol la."
        ),
    },
    "potato late blight": {
        "fr": "Mildiou de la pomme de terre",
        "bam": "Pomdetɛrɛ ni ji bana (mildiou)",
        "conseil_fr": (
            "• Traitement d'urgence avec un fongicide systémique.\n"
            "• Éliminez et brûlez toutes les parties malades.\n"
            "• Ne stockez pas les tubercules infectés.\n"
            "• Consultez un agent agricole immédiatement."
        ),
        "conseil_bam": (
            "• Buli fɛn joona joona.\n"
            "• Bɔ fɛnw minnu bana k'u jo.\n"
            "• Kana pomdetɛrɛ minnu bana mara.\n"
            "• Ɲinɛ wɛrɛ joona."
        ),
    },
    "potato healthy": {
        "fr": "Pomme de terre saine",
        "bam": "Pomdetɛrɛ ka kɛnɛya",
        "conseil_fr": "🥔 Votre pomme de terre est saine !",
        "conseil_bam": "🥔 I ka pomdetɛrɛ ni kɛnɛya. Kɔlɔsi a ka ɲɛ.",
    },

    # ── RAISIN ───────────────────────────────────────────────────────────
    "grape black rot": {
        "fr": "Pourriture noire du raisin",
        "bam": "Rɛzɛ ni finbana (noir)",
        "conseil_fr": (
            "• Retirez et détruisez les baies et feuilles infectées.\n"
            "• Appliquez un fongicide préventif (captan ou myclobutanil).\n"
            "• Assurez une bonne aération des vignes."
        ),
        "conseil_bam": (
            "• Bɔ fɛnw minnu bana k'u jo.\n"
            "• Buli fɛn ka ɲɛ.\n"
            "• Ka jaa kɛ ka diya."
        ),
    },
    "grape esca (black measles)": {
        "fr": "Esca (rougeot parasitaire) du raisin",
        "bam": "Rɛzɛ ni bana (esca)",
        "conseil_fr": (
            "• Aucun traitement chimique totalement efficace.\n"
            "• Taillez et brûlez les parties ligneuses infectées.\n"
            "• Protégez les plaies de taille avec une pâte cicatrisante.\n"
            "• Consultez un spécialiste viticole."
        ),
        "conseil_bam": (
            "• Bɔ fɛnw minnu bana k'u jo.\n"
            "• Bɔkɔlɔ bɔ k'a dɔgɔya.\n"
            "• Ɲinɛ wɛrɛ."
        ),
    },
    "grape leaf blight (isariopsis leaf spot)": {
        "fr": "Brûlure foliaire du raisin",
        "bam": "Rɛzɛ ni fɛn bana",
        "conseil_fr": (
            "• Appliquez un fongicide foliaire.\n"
            "• Retirez les feuilles très atteintes.\n"
            "• Améliorez la ventilation."
        ),
        "conseil_bam": (
            "• Buli fɛn kɛ fɛn kan.\n"
            "• Bɔ fɛnw minnu bana kosɛbɛ.\n"
            "• Ka jaa kɛ ka diya."
        ),
    },
    "grape healthy": {
        "fr": "Raisin sain",
        "bam": "Rɛzɛ ka kɛnɛya",
        "conseil_fr": "🍇 Votre vigne est saine !",
        "conseil_bam": "🍇 I ka rɛzɛ ni kɛnɛya.",
    },

    # ── POMME ────────────────────────────────────────────────────────────
    "apple apple scab": {
        "fr": "Tavelure du pommier",
        "bam": "Pomu ni bana (tavelure)",
        "conseil_fr": (
            "• Ramassez et détruisez les feuilles tombées.\n"
            "• Appliquez un fongicide dès le débourrement.\n"
            "• Choisissez des variétés résistantes."
        ),
        "conseil_bam": (
            "• Bɔ fɛnw minnu sera k'u jo.\n"
            "• Buli fɛn joona.\n"
            "• Sɛgɛsɛgɛ pomu wɛrɛ."
        ),
    },
    "apple black rot": {
        "fr": "Pourriture noire du pommier",
        "bam": "Pomu ni finbana",
        "conseil_fr": (
            "• Retirez les fruits et branches infectés.\n"
            "• Appliquez un fongicide cuivre.\n"
            "• Assurez une bonne taille de l'arbre."
        ),
        "conseil_bam": (
            "• Bɔ fɛnw minnu bana.\n"
            "• Buli fɛn (cuivre) kɛ.\n"
            "• Ka jiri bɔkɔlɔ bɔ."
        ),
    },
    "apple cedar apple rust": {
        "fr": "Rouille grillagée du pommier",
        "bam": "Pomu ni sɔgɔ bana",
        "conseil_fr": (
            "• Retirez les galles sur les genévriers à proximité.\n"
            "• Appliquez un fongicide au printemps.\n"
            "• Choisissez des variétés résistantes."
        ),
        "conseil_bam": (
            "• Bɔ fɛnw min bɛ jiri wɛrɛ kan.\n"
            "• Buli fɛn fɔlɔ saison la.\n"
            "• Sɛgɛsɛgɛ pomu wɛrɛ."
        ),
    },
    "apple healthy": {
        "fr": "Pommier sain",
        "bam": "Pomu ni kɛnɛya",
        "conseil_fr": "🍎 Votre pommier est sain !",
        "conseil_bam": "🍎 I ka pomu ni kɛnɛya.",
    },

    # ── CERISE ───────────────────────────────────────────────────────────
    "cherry (including sour) powdery mildew": {
        "fr": "Oïdium du cerisier",
        "bam": "Cerizi ni funubana (oïdium)",
        "conseil_fr": (
            "• Appliquez du soufre ou un fongicide anti-oïdium.\n"
            "• Améliorez la ventilation autour des arbres.\n"
            "• Évitez l'excès d'azote."
        ),
        "conseil_bam": (
            "• Buli sufuru wala buli fɛn kɛ.\n"
            "• Ka jaa kɛ ka diya.\n"
            "• Kana engrais caman kɛ."
        ),
    },
    "cherry (including sour) healthy": {
        "fr": "Cerisier sain",
        "bam": "Cerizi ni kɛnɛya",
        "conseil_fr": "🍒 Votre cerisier est sain !",
        "conseil_bam": "🍒 I ka cerizi ni kɛnɛya.",
    },

    # ── PÊCHE ────────────────────────────────────────────────────────────
    "peach bacterial spot": {
        "fr": "Tache bactérienne du pêcher",
        "bam": "Pɛci ni bana kɔrɔ",
        "conseil_fr": (
            "• Appliquez un fongicide cuivre au printemps.\n"
            "• Retirez les branches infectées.\n"
            "• Évitez les blessures mécaniques sur les fruits."
        ),
        "conseil_bam": (
            "• Buli fɛn (cuivre) fɔlɔ saison la.\n"
            "• Bɔ bɔkɔlɔ minnu bana.\n"
            "• Kana jelibaw kɛ."
        ),
    },
    "peach healthy": {
        "fr": "Pêcher sain",
        "bam": "Pɛci ni kɛnɛya",
        "conseil_fr": "🍑 Votre pêcher est sain !",
        "conseil_bam": "🍑 I ka pɛci ni kɛnɛya.",
    },

    # ── POIVRON ──────────────────────────────────────────────────────────
    "pepper, bell bacterial spot": {
        "fr": "Tache bactérienne du poivron",
        "bam": "Pimɛ ni bana kɔrɔ",
        "conseil_fr": (
            "• Appliquez un fongicide cuivre.\n"
            "• Évitez l'arrosage par aspersion.\n"
            "• Faites une rotation des cultures."
        ),
        "conseil_bam": (
            "• Buli fɛn (cuivre) kɛ.\n"
            "• Kana ji yɛlɛn kan bɔ.\n"
            "• Sannin taabol la."
        ),
    },
    "pepper, bell healthy": {
        "fr": "Poivron sain",
        "bam": "Pimɛ ni kɛnɛya",
        "conseil_fr": "🌶️ Votre poivron est sain !",
        "conseil_bam": "🌶️ I ka pimɛ ni kɛnɛya.",
    },

    # ── FRAISE ───────────────────────────────────────────────────────────
    "strawberry leaf scorch": {
        "fr": "Brûlure des feuilles du fraisier",
        "bam": "Fɛrɛzi ni fɛn bana",
        "conseil_fr": (
            "• Retirez les feuilles infectées.\n"
            "• Appliquez un fongicide approprié.\n"
            "• Évitez l'excès d'humidité."
        ),
        "conseil_bam": (
            "• Bɔ fɛnw minnu bana.\n"
            "• Buli fɛn kɛ.\n"
            "• Kana ji caman kɛ."
        ),
    },
    "strawberry healthy": {
        "fr": "Fraisier sain",
        "bam": "Fɛrɛzi ni kɛnɛya",
        "conseil_fr": "🍓 Votre fraisier est sain !",
        "conseil_bam": "🍓 I ka fɛrɛzi ni kɛnɛya.",
    },

    # ── SOJA ─────────────────────────────────────────────────────────────
    "soybean healthy": {
        "fr": "Soja sain",
        "bam": "Sojakisi ni kɛnɛya",
        "conseil_fr": "🫘 Votre plant de soja est sain !",
        "conseil_bam": "🫘 I ka sojakisi ni kɛnɛya.",
    },

    # ── COURGE / CITROUILLE ──────────────────────────────────────────────
    "squash powdery mildew": {
        "fr": "Oïdium de la courge",
        "bam": "Jɛgɛ ni funubana (oïdium)",
        "conseil_fr": (
            "• Appliquez du soufre ou du bicarbonate de soude dilué.\n"
            "• Améliorez la ventilation.\n"
            "• Évitez l'excès d'azote."
        ),
        "conseil_bam": (
            "• Buli sufuru wala soda kɛ.\n"
            "• Ka jaa kɛ ka diya.\n"
            "• Kana engrais caman kɛ."
        ),
    },

    # ── ORANGE ───────────────────────────────────────────────────────────
    "orange haunglongbing (citrus greening)": {
        "fr": "Huanglongbing (Greening des agrumes)",
        "bam": "Ɔransɛ ni bana gɛlɛn (greening)",
        "conseil_fr": (
            "• Aucun remède chimique efficace à ce jour.\n"
            "• Retirez et brûlez les arbres infectés.\n"
            "• Luttez contre le psylle (insecte vecteur).\n"
            "• Utilisez des plants certifiés sans maladie.\n"
            "• Signalez à l'autorité agricole locale."
        ),
        "conseil_bam": (
            "• Nin bana ma buli fɛn sɔrɔ.\n"
            "• Bɔ jiriw minnu bana k'u jo.\n"
            "• Buli fɛn kɛ kurunsi kan.\n"
            "• Sɛgɛsɛgɛ wari ka ɲɛ.\n"
            "• Kuma ni wɛrɛw ye."
        ),
    },

    # ── DÉFAUT (si la classe n'est pas dans le dictionnaire) ────────────
    "default": {
        "fr": "Maladie inconnue",
        "bam": "Bana tɔgɔ ma ɲɛfɔ",
        "conseil_fr": (
            "• Consultez un agent agricole local.\n"
            "• Prenez une photo claire et partagez-la.\n"
            "• Retirez les parties très atteintes."
        ),
        "conseil_bam": (
            "• Ɲinɛ wɛrɛ sɔrɔ.\n"
            "• Foto kɛ ka ɲɛ.\n"
            "• Bɔ fɛnw minnu bana kosɛbɛ."
        ),
    },
}


def get_disease_info(raw_label: str) -> dict:
    """
    Recherche les infos d'une maladie dans DISEASE_DB.
    raw_label peut être "Tomato___Late_blight" ou "tomato late blight".
    """
    # Normalisation : on retire le préfixe plante___ et on met en minuscules
    if "___" in raw_label:
        key = raw_label.split("___")[-1].replace("_", " ").lower()
        # Essai clé complète (plante + maladie)
        full_key = raw_label.replace("___", " ").replace("_", " ").lower()
    else:
        key = raw_label.replace("_", " ").lower()
        full_key = key

    for k in [full_key, key]:
        if k in DISEASE_DB:
            return DISEASE_DB[k]

    # Recherche partielle
    for db_key, val in DISEASE_DB.items():
        if db_key in full_key or full_key in db_key:
            return val

    return DISEASE_DB["default"]


# ─── AUDIO : gTTS (français) + MALIBA-AI (bambara) ───────────────────

def text_to_audio_b64_fr(text: str) -> str | None:
    """gTTS en français → base64 MP3."""
    try:
        tts = gTTS(text=text, lang="fr", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        st.warning(f"Audio français indisponible : {e}")
        return None


def text_to_audio_b64_bambara(text: str) -> str | None:
    """
    Appelle le Space HF MALIBA-AI/BambaraText2Speech via l'API Gradio.
    Retourne le base64 du WAV généré, ou None si le Space est indisponible.

    Le Space utilise maliba_ai sous Gradio — on interroge l'endpoint
    /predict avec le texte et l'ID du locuteur (Bourama par défaut).
    """
    try:
        from gradio_client import Client
        client = Client("MALIBA-AI/BambaraText2Speech", verbose=False)
        # L'interface Gradio expose : texte (str) + speaker (str)
        result = client.predict(
            text,
            "Bourama",   # locuteur masculin clair — idéal pour messages agricoles
            api_name="/predict",
        )
        # result est le chemin vers le fichier audio généré côté serveur
        if isinstance(result, str) and os.path.exists(result):
            with open(result, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        # Parfois result est un dict avec une clé "value" ou "path"
        if isinstance(result, dict):
            path = result.get("value") or result.get("path") or result.get("name")
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
        return None
    except Exception:
        # Space en veille ou gradio_client absent → fallback silencieux
        return None


def _audio_player_html(audio_b64: str, mime: str = "audio/mp3") -> str:
    return (
        f'<audio controls style="width:100%; margin-bottom:8px;">'
        f'<source src="data:{mime};base64,{audio_b64}" type="{mime}">'
        f"</audio>"
    )


def play_audio_fr(text: str, label: str = "Écouter en français 🇫🇷"):
    """Lecture audio en français via gTTS."""
    b64 = text_to_audio_b64_fr(text)
    if b64:
        st.markdown(f'<div class="audio-label">🔊 {label}</div>', unsafe_allow_html=True)
        st.markdown(_audio_player_html(b64, "audio/mp3"), unsafe_allow_html=True)


def play_audio_bambara(text: str, label: str = "Kelima bambara kan 🇲🇱"):
    """
    Lecture audio en bambara via MALIBA-AI/BambaraText2Speech.
    Affiche un spinner pendant la génération (le Space peut mettre 3-8s).
    Si le Space est indisponible, affiche un message discret sans planter.
    """
    with st.spinner("🔊 Génération audio bambara…"):
        b64 = text_to_audio_b64_bambara(text)

    if b64:
        st.markdown(f'<div class="audio-label">🔊 {label}</div>', unsafe_allow_html=True)
        # Le Space retourne du WAV
        st.markdown(_audio_player_html(b64, "audio/wav"), unsafe_allow_html=True)
    else:
        st.caption(
            "🔇 Audio bambara indisponible pour l'instant "
            "(Space en veille — réessayez dans 30 secondes)."
        )



# ─── CLASSES PLANTVILLAGE PAR DÉFAUT (38 classes) ─────────────────────
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot",
    "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]


def _build_model(num_classes: int) -> nn.Module:
    """Construit l'architecture EfficientNetV2-S (sans poids)."""
    backbone = timm.create_model(
        "tf_efficientnetv2_s.in21k",
        pretrained=False,
        num_classes=0,
        global_pool="avg",
    )
    in_features = backbone.num_features  # 1280

    class AgriMaliNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(in_features, 512),
                nn.SiLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes),
            )
        def forward(self, x):
            return self.classifier(self.backbone(x))

    return AgriMaliNet()


# ─── CHARGEMENT DU MODÈLE DEPUIS HUGGING FACE ─────────────────────────
@st.cache_resource
def load_model():
    """
    Télécharge agrimali_best-v2.pth depuis Abouba1810/AgriMali-EfficientNetV2
    et reconstruit le modèle EfficientNetV2-S.

    Priorité pour les classes :
      1. Clé "classes" dans le checkpoint (sauvegardée à l'entraînement)
      2. Fichier labels.txt local
      3. 38 classes PlantVillage standard
    """
    with st.spinner("📥 Chargement du modèle depuis Hugging Face…"):
        try:
            model_path = hf_hub_download(
                repo_id="sudoping01/crop-disease-detection",
                filename="best.pt",
                token=None,   # Mettre un token HF ici si le repo devient privé
            )
        except Exception as e:
            st.error(f"❌ Impossible de télécharger le modèle : {e}")
            return None, None

    # ── 1. Charger le checkpoint ────────────────────────────────────────
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        # weights_only non supporté sur les vieilles versions de PyTorch
        checkpoint = torch.load(model_path, map_location="cpu")

    # ── 2. Extraire state_dict et classes ──────────────────────────────
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model")
            or checkpoint.get("state_dict")
            or checkpoint
        )
        classes_from_ckpt = checkpoint.get("classes", None)
    else:
        state_dict = checkpoint
        classes_from_ckpt = None

    # ── 3. Résolution des classes (priorité : checkpoint > labels.txt > défaut) ─
    if classes_from_ckpt is not None and len(classes_from_ckpt) > 0:
        classes = list(classes_from_ckpt)
        st.success(f"✅ {len(classes)} classes chargées depuis le checkpoint.")
    elif os.path.exists("labels.txt"):
        with open("labels.txt", "r", encoding="utf-8") as f:
            classes = [l.strip() for l in f if l.strip()]
        st.success(f"✅ {len(classes)} classes chargées depuis labels.txt.")
    else:
        classes = PLANTVILLAGE_CLASSES
        st.info(f"ℹ️ Utilisation des {len(classes)} classes PlantVillage par défaut.")

    num_classes = len(classes)

    # ── 4. Construire le modèle et charger les poids ────────────────────
    model = _build_model(num_classes)

    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        st.error(f"❌ Erreur lors du chargement des poids : {e}")
        return None, None

    model.eval()
    return model, classes


# ─── RECADRAGE INTELLIGENT (OpenCV) ──────────────────────────────────
import cv2
import numpy as np

def smart_crop(image: Image.Image) -> tuple[Image.Image, bool]:
    """
    Détecte le plus grand contour vert/végétal (feuille) dans l'image
    et recadre autour de son bounding box avec une marge de 10%.

    Stratégie :
      1. Conversion en HSV → masque de la plage verte/végétale
      2. Morphologie pour nettoyer le masque
      3. Plus grand contour → bounding box
      4. Si la bbox couvre > 10% de l'image → recadrage
      5. Sinon fallback : crop carré centré (meilleur que rien)

    Retourne (image_croppée, crop_intelligent_réussi).
    """
    img_rgb = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w    = img_bgr.shape[:2]

    # ── 1. Masque HSV pour détecter les tons végétaux ─────────────────
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Plage verte (feuilles saines + malades jaunies/brûlées incluses)
    lower_green  = np.array([25,  30,  30])   # jaune-vert jusqu'à vert foncé
    upper_green  = np.array([95, 255, 255])
    mask_green   = cv2.inRange(hsv, lower_green, upper_green)

    # Plage brune/orange (taches de maladies fréquentes)
    lower_brown  = np.array([5,  40,  40])
    upper_brown  = np.array([25, 255, 220])
    mask_brown   = cv2.inRange(hsv, lower_brown, upper_brown)

    mask = cv2.bitwise_or(mask_green, mask_brown)

    # ── 2. Morphologie : fermeture pour boucher les trous ─────────────
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # ── 3. Trouver le plus grand contour ──────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return _center_square_crop(image), False

    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)

    # Si le contour est trop petit (< 10% de l'image) → pas de feuille détectée
    if area < 0.10 * h * w:
        return _center_square_crop(image), False

    # ── 4. Bounding box + marge 10% ───────────────────────────────────
    x, y, bw, bh = cv2.boundingRect(largest)
    margin_x = int(bw * 0.10)
    margin_y = int(bh * 0.10)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(w, x + bw + margin_x)
    y2 = min(h, y + bh + margin_y)

    # ── 5. Recadrage carré centré sur la bbox (évite déformation) ─────
    crop_w = x2 - x1
    crop_h = y2 - y1
    side   = max(crop_w, crop_h)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    sx = max(0, cx - side // 2)
    sy = max(0, cy - side // 2)
    ex = min(w, sx + side)
    ey = min(h, sy + side)

    cropped_bgr = img_bgr[sy:ey, sx:ex]
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))
    return cropped_pil, True


def _center_square_crop(image: Image.Image) -> Image.Image:
    """Crop carré centré — fallback simple."""
    w, h      = image.size
    side      = min(w, h)
    left      = (w - side) // 2
    top       = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


# ─── INFÉRENCE ────────────────────────────────────────────────────────
def _run_inference(image: Image.Image, model, classes) -> list:
    """Inférence brute sur une image PIL."""
    transform = transforms.Compose([
        transforms.Resize((int(384 * 1.14), int(384 * 1.14))),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs       = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top3_probs, top3_idx = torch.topk(probabilities, min(3, len(classes)))
    results = []
    for prob, idx in zip(top3_probs.tolist(), top3_idx.tolist()):
        raw   = classes[idx]
        clean = (raw.split("___")[-1].replace("_", " ").title()
                 if "___" in raw else raw.replace("_", " ").title())
        results.append({
            "label":      clean,
            "raw":        raw,
            "confidence": prob * 100,
            "is_healthy": "healthy" in raw.lower(),
            "info":       get_disease_info(raw),
        })
    return results


CONFIDENCE_THRESHOLD = 55.0   # en dessous : analyse incertaine
SMART_CROP_TRIGGER   = 70.0   # en dessous : on tente le recadrage intelligent


def predict(image: Image.Image, model, classes):
    """
    Stratégie en 2 passes :
      1. Inférence sur l'image originale.
      2. Si confiance < SMART_CROP_TRIGGER, on détecte et recadre
         la feuille avec OpenCV, on retente et on garde le meilleur.

    Retourne (results, crop_used: bool, image_used: PIL.Image).
    """
    results_orig = _run_inference(image, model, classes)
    top_conf     = results_orig[0]["confidence"]

    if top_conf >= SMART_CROP_TRIGGER:
        return results_orig, False, image

    # 2ème passe avec recadrage intelligent
    image_cropped, crop_succeeded = smart_crop(image)

    if not crop_succeeded:
        # Pas de feuille détectée → inutile de re-inférer
        return results_orig, False, image

    results_crop  = _run_inference(image_cropped, model, classes)
    top_conf_crop = results_crop[0]["confidence"]

    if top_conf_crop > top_conf:
        return results_crop, True, image_cropped
    else:
        return results_orig, False, image


# ─── INTERFACE ────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center; color:#2e7d32; font-family:sans-serif;'>🌿 Agri'Mali</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#666; margin-top:-15px;'>Votre assistant agricole intelligent</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

model, classes = load_model()
if model is None:
    st.stop()

st.markdown(
    '<div class="hint-text">📸 Prenez une photo claire de la feuille de votre plante, '
    'ou importez une image depuis votre galerie.</div>',
    unsafe_allow_html=True,
)

# ─── ONGLETS PHOTO / GALERIE ─────────────────────────────────────────
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

# ─── ANALYSE ─────────────────────────────────────────────────────────
active_file   = st.session_state.get("active_image", None)
image_source  = st.session_state.get("image_source", None)

if active_file is not None:
    try:
        image = Image.open(active_file).convert("RGB")
    except Exception as e:
        st.error(f"❌ Impossible d'ouvrir l'image : {e}")
        st.stop()

    if image_source == "gallery":
        st.image(image, caption="Image sélectionnée", use_container_width=True)

    st.write("")
    if st.button("🔍 ANALYSER LA FEUILLE"):
        with st.spinner("🔬 Analyse en cours…"):
            results, crop_was_used, image_used = predict(image, model, classes)

        # ── Affichage comparatif si le recadrage a amélioré ──────────
        if crop_was_used:
            st.markdown(
                "<p style='font-size:13px; color:#2e7d32; margin-bottom:4px;'>"
                "✂️ <b>Recadrage automatique</b> — feuille détectée et centrée "
                "pour une meilleure analyse.</p>",
                unsafe_allow_html=True,
            )
            col1, col2 = st.columns(2)
            with col1:
                st.image(image,      caption="Photo originale",      use_container_width=True)
            with col2:
                st.image(image_used, caption="Feuille recadrée (analysée)", use_container_width=True)

        top  = results[0]
        info = top["info"]
        st.write("### Résultat")

        # ── Cas 1 : Faible confiance ─────────────────────────────────
        if top["confidence"] < 55.0:
            st.markdown(f"""
                <div style="background-color:#ff9800; color:white; padding:25px;
                            border-radius:16px; text-align:center;
                            box-shadow:0px 4px 12px rgba(255,152,0,0.35);">
                    <div style='font-size:50px;'>🤔</div>
                    <h2 style='margin:5px 0 0 0;'>ANALYSE INCERTAINE</h2>
                    <h3 style='margin:8px 0 0 0;'>Confiance : {top['confidence']:.1f}%</h3>
                    <p style='margin:10px 0 0 0; opacity:0.95;'>
                        L'IA n'arrive pas à identifier clairement l'état de cette feuille.
                    </p>
                    <hr style='border:1px solid rgba(255,255,255,0.4); margin:15px 0;'>
                    <p style='font-style:italic; margin:0; font-size:14px;'>
                        💡 Conseils photo : rapprochez-vous, utilisez la lumière naturelle,
                        évitez les ombres, cadrez une seule feuille.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            play_audio_fr(
                "Analyse incertaine. Veuillez reprendre une photo plus claire "
                "avec une bonne lumière naturelle et une seule feuille dans le cadre.",
            )
            play_audio_bambara(
                "Sɛgɛsɛgɛ kɛra ka gɛlɛn. Foto wɛrɛ kɛ ka ɲɛ, "
                "ni yeelen ka ɲɛ, fɛn kelen dɔrɔn."
            )

        # ── Cas 2 : Plante saine ─────────────────────────────────────
        elif top["is_healthy"]:
            nom_fr  = info["fr"]
            nom_bam = info["bam"]
            st.markdown(f"""
                <div class="status-card-healthy">
                    <div style='font-size:50px;'>😊</div>
                    <h2 style='margin:5px 0 0 0;'>PLANTE SAINE</h2>
                    <p style='margin:5px 0 0 0; opacity:0.9;'>{nom_fr}</p>
                    <h3 style='margin:10px 0 0 0;'>Confiance : {top['confidence']:.0f}%</h3>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(
                f'<div class="bambara-box">🗣️ <b>Bambara :</b> {nom_bam}<br><br>'
                f'{info["conseil_bam"]}</div>',
                unsafe_allow_html=True,
            )
            st.balloons()
            st.info(f"🌿 **{nom_fr}**\n\n{info['conseil_fr']}")

            play_audio_fr(f"{nom_fr}. {info['conseil_fr']}")
            play_audio_bambara(f"{nom_bam}. {info['conseil_bam']}")

        # ── Cas 3 : Maladie détectée ─────────────────────────────────
        else:
            nom_fr  = info["fr"]
            nom_bam = info["bam"]
            st.markdown(f"""
                <div class="status-card-sick">
                    <div style='font-size:50px;'>🙁</div>
                    <h2 style='margin:5px 0 0 0;'>PLANTE MALADE</h2>
                    <p style='margin:5px 0 0 0; opacity:0.9;'>{nom_fr}</p>
                    <h3 style='margin:10px 0 0 0;'>Confiance : {top['confidence']:.0f}%</h3>
                </div>
            """, unsafe_allow_html=True)

            st.markdown(
                f'<div class="bambara-box">🗣️ <b>Bambara :</b> {nom_bam}<br><br>'
                f'<b>Ɲɛsira (Conseils) :</b><br>{info["conseil_bam"]}</div>',
                unsafe_allow_html=True,
            )
            st.error(f"**Maladie : {nom_fr}**")
            st.warning(f"**Que faire ?**\n\n{info['conseil_fr']}")

            play_audio_fr(f"Maladie détectée : {nom_fr}. {info['conseil_fr']}")
            play_audio_bambara(f"Bana tɔgɔ : {nom_bam}. {info['conseil_bam']}")

        # ── Top 3 ───────────────────────────────────────────────────
        if len(results) > 1:
            st.write("### 📊 Top 3 des prédictions")
            for i, r in enumerate(results):
                emoji = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")
                color = "#2e7d32" if r["is_healthy"] else "#d32f2f"
                bar_width = int(r["confidence"])
                label_display = r["info"]["fr"] if r["info"] else r["label"]
                st.markdown(f"**{emoji} {label_display}** — {r['confidence']:.1f}%")
                st.markdown(
                    f'<div class="confidence-bar-bg">'
                    f'<div style="background:{color};height:12px;border-radius:8px;width:{bar_width}%"></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

else:
    st.markdown(
        "<p style='text-align:center; color:#aaa; margin-top:30px;'>📷 Aucune image sélectionnée</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:12px;'>"
    "Agri'Mali — Développé pour les agriculteurs maliens 🇲🇱</p>",
    unsafe_allow_html=True,
)
