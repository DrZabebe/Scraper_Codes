

import requests

# Your SerpApi API Key
API_KEY = 'e141e92414ffd37ff53bf7f7426139bd2d6d86ad8daaf6ac569bd9f62f56302b'

# List of journals to filter results for
target_journals = [
    "IEEE Transactions on Information Technology in Biomedicine",
    "Journal of Information Technology",
    "Journal of Management Information Systems (JMIS)",
    "ACM Transactions on Information Systems (TOIS)",
    "Information Systems Research (ISR)",
    "Journal of the Association for Information Systems (JAIS)",
    "MIS Quarterly (Management Information Systems Quarterly)",
    "Information and Management",
    "Computers in Human Behavior",
    "Journal of Computer-Mediated Communication",
    "Computer Networks",
    "Software: Practice and Experience",
    "International Journal of Information Management",
    "Journal of Strategic Information Systems",
    "Electronic Commerce Research and Applications",
    "International Journal of Computer Science and Information Security (IJCSIS)",
    "Information Technology and Management",
    "Information Systems Journal",
    "International Journal of Information Technology and Decision Making (IJITDM)",
    "ACM Computing Surveys"
]

# List of search queries based on your topics
# Comprehensive list of search queries focusing on human-related barriers to AI adoption in healthcare
queries = [
    "Human-related barriers to AI adoption in healthcare information systems",
    "Challenges faced by healthcare professionals in adopting AI technologies",
    "Trust, usability, and impact on patient safety in AI adoption by healthcare providers",
    "Organizational factors hindering AI adoption in healthcare systems",
    "Training, cost, and system readiness as barriers to AI adoption in healthcare",
    "Strategies to overcome AI adoption barriers in under-resourced healthcare settings",
    "Perceptions of AI among healthcare professionals and decision-makers",
    "Cultural and social factors affecting AI adoption in healthcare",
    "Impact of regulatory policies on AI implementation in healthcare",
    "Data quality and interoperability challenges in AI adoption",
    "Ethical considerations in AI adoption within healthcare",
    "Patient acceptance and trust in AI-driven healthcare solutions",
    "Integration of AI into clinical workflows and its challenges",
    "Evaluating the effectiveness of AI interventions in healthcare settings",
    "Training and education requirements for successful AI adoption in healthcare",
    "Cost-benefit analysis of implementing AI in healthcare organizations",
    "Case studies on overcoming AI adoption challenges in healthcare",
    "Future trends and predictions for AI adoption in healthcare",
    "Algorithm aversion and strategies to mitigate resistance to AI in healthcare",
    "Human-in-the-loop approaches to enhance AI acceptance in clinical settings",
    "Transparency and explainability in AI systems to build trust among healthcare providers",
    "User training programs to reduce algorithm aversion in healthcare",
    "Incorporating user control to improve acceptance of AI recommendations in healthcare",
    "Personalization and customization of AI systems to align with healthcare providers' needs",
    "Addressing ethical challenges such as bias, accountability, and privacy violations in AI adoption",
    "Building trust in AI through education and accreditation of medical AI experts",
    "Overcoming resistance to AI adoption through organizational change management",
    "Leveraging influential staff to promote AI adoption in healthcare settings",
    "Addressing data privacy concerns to enhance trust in AI systems among patients and providers",
    "Developing policies to support ethical and effective AI implementation in healthcare",
    "Evaluating the impact of AI on patient outcomes to justify its adoption",
    "Balancing technological advancement with human touch in AI-assisted patient care",
    "Ensuring equitable access to AI technologies across different healthcare settings",
    "Assessing the role of AI in reducing healthcare disparities and improving patient care",
    "Understanding clinician acceptance of AI-based treatment recommendations",
    "Addressing algorithm aversion to improve acceptance of AI in healthcare",
    "Integrating AI into existing healthcare infrastructures to enhance usability",
    "Evaluating the impact of AI on patient outcomes to justify its adoption",
    "Balancing technological advancement with human touch in AI-assisted patient care",
    "Ensuring equitable access to AI technologies across different healthcare settings",
    "Assessing the role of AI in reducing healthcare disparities and improving patient care",
    "Understanding clinician acceptance of AI-based treatment recommendations",
    "Addressing algorithm aversion to improve acceptance of AI in healthcare",
    "Integrating AI into existing healthcare infrastructures to enhance usability",
    "Strategies to overcome algorithm aversion in healthcare AI adoption",
    "Human-in-the-loop approaches to mitigate resistance to AI in healthcare",
    "Enhancing transparency and explainability to build trust in AI systems among healthcare providers",
    "User training programs to familiarize healthcare professionals with AI tools",
    "Incorporating user control features in AI systems to improve acceptance among clinicians",
    "Personalization of AI recommendations to align with individual clinician preferences",
    "Addressing ethical concerns such as bias and accountability in AI algorithms",
    "Establishing governance frameworks to oversee AI implementation in healthcare",
    "Building trust in AI through education and accreditation of medical AI experts",
    "Overcoming resistance to AI adoption through effective change management strategies",
    "Leveraging influential staff to champion AI adoption within healthcare organizations",
    "Addressing data privacy concerns to enhance trust in AI systems among patients and providers",
    "Developing policies to support ethical and effective AI implementation in healthcare",
    "Evaluating the impact of AI on patient outcomes to justify its adoption",
    "Balancing technological advancement with human touch in AI-assisted patient care",
    "Ensuring equitable access to AI technologies across different healthcare settings",
    "Assessing the role of AI in reducing healthcare disparities and improving patient care",
    "Understanding clinician acceptance of AI-based treatment recommendations",
    "Addressing algorithm aversion to improve acceptance of AI in healthcare",
    "Integrating AI into existing healthcare infrastructures to enhance usability",
    "Strategies to overcome algorithm aversion in healthcare AI adoption",
    "Human-in-the-loop approaches to mitigate resistance to AI in healthcare",
    "Enhancing transparency and explainability to build trust in AI systems among healthcare providers",
    "User training programs to familiarize healthcare professionals with AI tools",
    "Incorporating user control features in AI systems to improve acceptance among clinicians",
    "Personalization of AI recommendations to align with individual clinician preferences",
    "Addressing ethical concerns such as bias and accountability in AI algorithms",
    "Establishing governance frameworks to oversee AI implementation in healthcare",
    "Building trust in AI through education and accreditation of medical AI experts",
    "Overcoming resistance to AI adoption through effective change management strategies",
    "Leveraging influential staff to champion AI adoption within healthcare organizations",
    "Addressing data privacy concerns to enhance trust in AI systems among patients and providers",
    "Developing policies to support ethical and effective AI implementation in healthcare",
    "Evaluating the impact of AI on patient outcomes to justify its adoption",
    "Balancing technological advancement with human touch in AI-assisted patient care",
    "Ensuring equitable access to AI technologies across different healthcare settings",
    "Assessing the role of AI in reducing healthcare disparities and improving patient care"
]

# Function to fetch search results from Google Scholar via SerpApi
def search_scholar(query):
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": API_KEY,
        "num": 5,  # Number of results per query
    }
    response = requests.get("https://serpapi.com/search", params=params)
    results = response.json().get("organic_results", [])

    print(f"\n🔍 Results for: {query}\n")
    for result in results:
        title = result.get("title")
        link = result.get("link")
        publication = result.get("publication_info", {}).get("summary")

        # Check if any of the target journals match the result
        if any(journal in publication for journal in target_journals):
            print(f"• {title}\n  {publication}\n  {link}\n")


# Run search for all queries
for q in queries:
    search_scholar(q)