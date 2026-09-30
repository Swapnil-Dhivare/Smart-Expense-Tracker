"""ML-driven expense categorization.

Every prediction goes through a trained classifier — there is no hardcoded
merchant->category lookup table. The pipeline is:

  1. normalize_text(): strip real bank-statement noise (UPI/POS/NEFT
     prefixes, payment-handle suffixes, reference numbers) — feature
     engineering, not classification.
  2. TF-IDF (word n-grams) -> Logistic Regression, trained on labeled
     merchant names plus generic descriptive phrases so it can generalize
     to merchants it has never seen.

Cross-validation accuracy is computed and exposed (`cv_accuracy_mean/std`,
`confusion_matrix_`, `classification_report_`) for the Model Insights view.
A correction fed back through `retrain()` is folded into the training set
and the model is refit — a small active-learning loop.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

CATEGORIES = [
    "Food",
    "Groceries",
    "Transport",
    "Shopping",
    "Entertainment",
    "Bills & Utilities",
    "Health",
    "Travel",
    "Education",
    "Others",
]

# Training data: (merchant/description text, category)
TRAINING_DATA: list[tuple[str, str]] = [
    # Food
    ("Swiggy", "Food"), ("Zomato", "Food"), ("Swiggy Food Order", "Food"),
    ("McDonalds", "Food"), ("Dominos Pizza", "Food"), ("KFC", "Food"),
    ("Starbucks", "Food"), ("Burger King", "Food"), ("Pizza Hut", "Food"),
    ("Subway", "Food"), ("Cafe Coffee Day", "Food"), ("Barbeque Nation", "Food"),
    ("Haldirams", "Food"), ("Chai Point", "Food"), ("Behrouz Biryani", "Food"),
    ("EatSure", "Food"), ("Faasos", "Food"), ("Local Restaurant", "Food"),
    ("Cafe Coffee", "Food"), ("Wow Momo", "Food"), ("Third Wave Coffee", "Food"),
    # Groceries
    ("BigBasket", "Groceries"), ("Blinkit", "Groceries"), ("Zepto", "Groceries"),
    ("Grofers", "Groceries"), ("Dmart", "Groceries"), ("Reliance Fresh", "Groceries"),
    ("More Supermarket", "Groceries"), ("JioMart Grocery", "Groceries"),
    ("Local Kirana Store", "Groceries"), ("Nature Basket", "Groceries"),
    ("Spencers", "Groceries"), ("Star Bazaar", "Groceries"),
    # Transport
    ("Uber", "Transport"), ("Ola Cabs", "Transport"), ("Rapido", "Transport"),
    ("Indian Railways", "Transport"), ("IRCTC", "Transport"), ("Delhi Metro", "Transport"),
    ("HP Petrol Pump", "Transport"), ("Indian Oil", "Transport"), ("Shell Petrol", "Transport"),
    ("Parking Fee", "Transport"), ("FASTag Toll", "Transport"), ("Bounce Scooter", "Transport"),
    ("Yulu Bike", "Transport"), ("Auto Rickshaw", "Transport"), ("BluSmart", "Transport"),
    # Shopping
    ("Amazon", "Shopping"), ("Flipkart", "Shopping"), ("Myntra", "Shopping"),
    ("Ajio", "Shopping"), ("Nykaa", "Shopping"), ("Meesho", "Shopping"),
    ("Snapdeal", "Shopping"), ("Big Bazaar", "Shopping"), ("Reliance Trends", "Shopping"),
    ("Zara", "Shopping"), ("H&M", "Shopping"), ("Lifestyle Store", "Shopping"),
    ("Decathlon", "Shopping"), ("Croma", "Shopping"), ("IKEA", "Shopping"),
    ("Pantaloons", "Shopping"), ("Shoppers Stop", "Shopping"), ("Amazon Fashion", "Shopping"),
    # Entertainment
    ("Netflix", "Entertainment"), ("Amazon Prime Video", "Entertainment"),
    ("Hotstar", "Entertainment"), ("Spotify", "Entertainment"), ("BookMyShow", "Entertainment"),
    ("PVR Cinemas", "Entertainment"), ("INOX", "Entertainment"), ("Gaana", "Entertainment"),
    ("YouTube Premium", "Entertainment"), ("JioCinema", "Entertainment"),
    ("SonyLIV", "Entertainment"), ("Steam Games", "Entertainment"), ("PlayStation Store", "Entertainment"),
    # Bills & Utilities
    ("Electricity Board", "Bills & Utilities"), ("Airtel Postpaid", "Bills & Utilities"),
    ("Jio Recharge", "Bills & Utilities"), ("Vodafone Idea", "Bills & Utilities"),
    ("BSNL Bill", "Bills & Utilities"), ("Water Bill", "Bills & Utilities"),
    ("Gas Cylinder Booking", "Bills & Utilities"), ("ACT Broadband", "Bills & Utilities"),
    ("Tata Sky DTH", "Bills & Utilities"), ("Piped Gas Bill", "Bills & Utilities"),
    ("Society Maintenance", "Bills & Utilities"), ("Mobile Recharge", "Bills & Utilities"),
    # Health
    ("Apollo Pharmacy", "Health"), ("Practo Consultation", "Health"), ("Medplus", "Health"),
    ("Fortis Hospital", "Health"), ("City Clinic", "Health"), ("Cult Fit", "Health"),
    ("Health Insurance Premium", "Health"), ("Netmeds", "Health"), ("1mg Pharmacy", "Health"),
    ("Gym Membership", "Health"), ("Diagnostic Lab", "Health"), ("Dentist", "Health"),
    # Travel
    ("MakeMyTrip", "Travel"), ("Goibibo", "Travel"), ("IndiGo Airlines", "Travel"),
    ("SpiceJet", "Travel"), ("Air India", "Travel"), ("OYO Rooms", "Travel"),
    ("Airbnb", "Travel"), ("Yatra", "Travel"), ("Cleartrip", "Travel"),
    ("Hotel Booking", "Travel"), ("Vistara Airlines", "Travel"),
    # Education
    ("Udemy Course", "Education"), ("Coursera", "Education"), ("BYJUS", "Education"),
    ("Unacademy", "Education"), ("School Fee", "Education"), ("Tuition Fee", "Education"),
    ("College Fee", "Education"), ("upGrad", "Education"), ("Skillshare", "Education"),
    # Others
    ("ATM Withdrawal", "Others"), ("Cash Withdrawal", "Others"), ("Bank Charges", "Others"),
    ("Fund Transfer", "Others"), ("Miscellaneous", "Others"), ("Donation", "Others"),
    ("Insurance Premium", "Others"), ("Credit Card Payment", "Others"), ("Rent Payment", "Others"),
]

# Generic descriptive phrases (not brand names) so the classifier can
# generalize to merchants it has never seen, via shared vocabulary rather
# than memorized brand names.
DESCRIPTIVE_TRAINING_DATA: list[tuple[str, str]] = [
    ("restaurant bill", "Food"), ("dining out", "Food"), ("food order", "Food"),
    ("cafe bill", "Food"), ("lunch order", "Food"), ("dinner order", "Food"),
    ("grocery shopping", "Groceries"), ("supermarket purchase", "Groceries"),
    ("vegetable market", "Groceries"), ("kirana store purchase", "Groceries"),
    ("cab fare", "Transport"), ("taxi fare", "Transport"), ("fuel purchase", "Transport"),
    ("petrol filling", "Transport"), ("bus ticket", "Transport"), ("train ticket", "Transport"),
    ("toll payment", "Transport"), ("parking charge", "Transport"),
    ("online shopping", "Shopping"), ("clothing purchase", "Shopping"),
    ("electronics purchase", "Shopping"), ("furniture purchase", "Shopping"),
    ("footwear purchase", "Shopping"),
    ("movie ticket", "Entertainment"), ("concert ticket", "Entertainment"),
    ("streaming subscription", "Entertainment"), ("music subscription", "Entertainment"),
    ("game purchase", "Entertainment"),
    ("electricity bill payment", "Bills & Utilities"), ("internet bill", "Bills & Utilities"),
    ("phone bill payment", "Bills & Utilities"), ("water bill payment", "Bills & Utilities"),
    ("gas bill payment", "Bills & Utilities"), ("maintenance charge", "Bills & Utilities"),
    ("doctor visit", "Health"), ("pharmacy purchase", "Health"), ("hospital bill", "Health"),
    ("medical checkup", "Health"), ("gym fees", "Health"), ("health checkup", "Health"),
    ("flight booking", "Travel"), ("hotel stay", "Travel"), ("train booking", "Travel"),
    ("vacation package", "Travel"), ("holiday booking", "Travel"),
    ("course fee", "Education"), ("tuition payment", "Education"),
    ("school fee payment", "Education"), ("online course purchase", "Education"),
    ("exam fee", "Education"),
    ("bank transfer", "Others"), ("loan emi payment", "Others"), ("insurance payment", "Others"),
    ("miscellaneous expense", "Others"),
]

_CONFIDENCE_FLOOR = 0.35

# Real bank-statement noise: UPI/POS/NEFT/IMPS prefixes, payment-handle
# suffixes (@okaxis, @ybl, ...), and reference numbers.
_NOISE_PATTERNS = [
    (re.compile(r"\bupi\b[-/]?"), " "),
    (re.compile(r"\bpos\b"), " "),
    (re.compile(r"\bneft\b[-/]?"), " "),
    (re.compile(r"\bimps\b[-/]?"), " "),
    (re.compile(r"\bach\b[-/]?"), " "),
    (re.compile(r"\bnach\b[-/]?"), " "),
    (re.compile(r"\bvpa\b[-/]?"), " "),
    (re.compile(r"\bref\b\.?"), " "),
    (re.compile(r"\btxn\b\.?"), " "),
    (re.compile(r"@[a-z]+"), " "),
    (re.compile(r"\d{4,}"), " "),
    (re.compile(r"[-_/*]+"), " "),
    (re.compile(r"\s{2,}"), " "),
]


def normalize_text(text: str) -> str:
    t = text.lower()
    for pattern, repl in _NOISE_PATTERNS:
        t = pattern.sub(repl, t)
    return t.strip()


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True)),
        ("clf", LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced")),
    ])


class ExpenseCategorizer:
    def __init__(self) -> None:
        self._extra_examples: list[tuple[str, str]] = []
        self._fit()

    def _training_set(self) -> tuple[list[str], list[str]]:
        combined = TRAINING_DATA + DESCRIPTIVE_TRAINING_DATA + self._extra_examples
        texts = [normalize_text(t) for t, _ in combined]
        labels = [c for _, c in combined]
        return texts, labels

    def _fit(self) -> None:
        texts, labels = self._training_set()
        self._pipeline = _build_pipeline()
        self._pipeline.fit(texts, labels)
        self._evaluate(texts, labels)

    def _evaluate(self, texts: list[str], labels: list[str]) -> None:
        label_counts = {c: labels.count(c) for c in set(labels)}
        min_class_count = min(label_counts.values())
        n_splits = min(5, min_class_count)
        if n_splits < 2:
            self.cv_accuracy_mean = None
            self.cv_accuracy_std = None
            self.confusion_matrix_ = None
            self.classification_report_ = None
            self.cv_labels_ = sorted(set(labels))
            return

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(_build_pipeline(), texts, labels, cv=skf)
        self.cv_accuracy_mean = float(scores.mean())
        self.cv_accuracy_std = float(scores.std())

        preds = cross_val_predict(_build_pipeline(), texts, labels, cv=skf)
        self.cv_labels_ = sorted(set(labels))
        self.confusion_matrix_ = confusion_matrix(labels, preds, labels=self.cv_labels_)
        self.classification_report_ = classification_report(
            labels, preds, labels=self.cv_labels_, output_dict=True, zero_division=0
        )

    def retrain(self, extra_examples: list[tuple[str, str]]) -> None:
        """Fold corrected examples into the training set and refit."""
        self._extra_examples.extend(extra_examples)
        self._fit()

    @property
    def correction_count(self) -> int:
        return len(self._extra_examples)

    @property
    def training_example_count(self) -> int:
        return len(TRAINING_DATA) + len(DESCRIPTIVE_TRAINING_DATA) + len(self._extra_examples)

    def top_tokens(self, category: str, top_n: int = 8) -> list[tuple[str, float]]:
        """The n-grams the model weighs most heavily for a category."""
        vectorizer: TfidfVectorizer = self._pipeline.named_steps["tfidf"]
        clf: LogisticRegression = self._pipeline.named_steps["clf"]
        if category not in clf.classes_:
            return []
        class_idx = list(clf.classes_).index(category)
        coefs = clf.coef_[class_idx] if len(clf.classes_) > 2 else clf.coef_[0]
        feature_names = vectorizer.get_feature_names_out()
        top_idx = np.argsort(coefs)[::-1][:top_n]
        return [(feature_names[i], float(coefs[i])) for i in top_idx if coefs[i] > 0]

    def predict(self, descriptions: list[str]) -> list[tuple[str, float]]:
        """Return (category, confidence) for each description."""
        normalized = [normalize_text(d) for d in descriptions]
        probs = self._pipeline.predict_proba(normalized)
        classes = self._pipeline.classes_
        results = []
        for row in probs:
            best_idx = int(np.argmax(row))
            category = classes[best_idx]
            confidence = float(row[best_idx])
            if confidence < _CONFIDENCE_FLOOR:
                category = "Others"
            results.append((category, confidence))
        return results
