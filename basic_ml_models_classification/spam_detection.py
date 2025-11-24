import pandas as pd

def spam_types():
    data = {
        "Category": [
            "By Approach", "", "",
            "By Learning Paradigm", "", "",
            "By Communication Channel", "", "",
            "By Detection Technique", "", "", "", ""
        ],
        "Type": [
            "Rule-Based", "Machine Learning", "Hybrid",
            "Supervised Learning", "Unsupervised Learning", "Semi-Supervised Learning",
            "Email Spam Detection", "SMS Spam Detection", "Social Media Spam Detection",
            "Content-Based Filtering", "Sender-Based Filtering", "Behavioral Detection",
            "Image-Based Spam Detection", "Network-Based Detection"
        ],
        "Key Characteristic": [
            "Uses manually created rules (keywords, patterns, blacklists)",
            "Learns hidden patterns from past spam/ham data",
            "Combines ML + rules + heuristics",
            "Uses labeled dataset (spam, ham)",
            "Finds clusters/anomalies without labels",
            "Mix of labeled + unlabeled data",
            "Looks at body, subject, headers, sender domain",
            "Analyzes short messages, URLs, phone numbers",
            "Analyzes posts, comments, behavior",
            "Focuses on message text, links, metadata",
            "Checks sender reputation, IP/domain blacklists",
            "Detects abnormal posting or sending patterns",
            "OCR + CNN for detecting spam inside images",
            "Traffic analysis (message volume, patterns)"
        ],
        "Best For": [
            "Simple, first-pass filtering",
            "Adaptive, large-scale systems",
            "Strong, multi-layered spam defense (Gmail-type filters)",
            "High accuracy, production systems",
            "Detecting new, unknown spam patterns",
            "Real-world cases with limited labeled data",
            "Email systems (Gmail, Outlook)",
            "Telecom, banking OTP protection",
            "Instagram, YouTube, Facebook moderation",
            "Email/SMS text spam",
            "Email filtering systems",
            "Bots, fake accounts, mass posting",
            "Image-based email spam",
            "Telecom, ISP-level spam detection"
        ]
    }
    df = pd.DataFrame(data)
    return df
