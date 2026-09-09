import os
import pandas as pd
import numpy as np

def generate_sentiment_dataset():
    os.makedirs('data', exist_ok=True)
    os.makedirs('images', exist_ok=True)

    positive_samples = [
        "I absolutely love this product! It exceeded all my expectations and works perfectly.",
        "Fantastic customer service! The team was super helpful and resolved my issue immediately.",
        "Great battery life, sleek design, and blazingly fast performance. Highly recommended!",
        "The meal was delicious, the ambiance was cozy, and the staff were exceptionally polite.",
        "Outstanding movie! The acting was superb and the plot kept me engaged from start to finish.",
        "This flight was on time, super smooth, and the flight attendants were extremely courteous.",
        "Best purchase I've made all year. High quality materials and worth every penny.",
        "The software update made everything so smooth and intuitive. Incredible upgrade!",
        "Very satisfied with the speedy delivery and excellent packaging. 10/10 service!",
        "Amazing concert experience! The sound quality and stage lighting were spectacular.",
        "Super comfortable shoes for running and daily workouts. Fits like a glove!",
        "The camera quality on this phone is unreal. Daylight and night photos look professional.",
        "Wonderful experience staying at this hotel. Rooms were spotless and staff was friendly.",
        "The user interface is clean, beautiful, and easy to navigate. Kudos to the design team!",
        "Highly impressed by the durability and build quality of this laptop stand.",
        "Fast shipping, great communication, and product works exactly as advertised.",
        "The customer support rep went above and beyond to answer all my detailed questions.",
        "Delightful dining experience with incredible flavors and top-tier presentation.",
        "Extremely pleased with my purchase. Will definitely recommend to all my friends!",
        "The noise cancellation on these headphones is top-notch. Blocks out everything."
    ]

    negative_samples = [
        "Terrible experience! The product broke within two days of normal usage. Total waste of money.",
        "Horrible customer service. Spent 45 minutes on hold and got disconnected without any resolution.",
        "Extremely disappointing quality. The item looks cheap and nothing like the photos online.",
        "The food was cold, tasteless, and arrived over an hour late. Never ordering again.",
        "Awful movie with terrible dialogue, poor pacing, and flat acting throughout.",
        "Flight was delayed for three hours without any explanation or compensation. Very frustrated!",
        "Worst customer support ever. The agent was rude and refused to honor the warranty.",
        "The latest update completely ruined the application. Crashes every time I open it.",
        "Item arrived damaged with missing parts. Return process has been a absolute nightmare.",
        "Overpriced and underperforming. Battery dies in less than two hours. Do not buy!",
        "Uncomfortable seats and noisy environment. Slept horribly during my stay.",
        "The app is laggy, full of bugs, and keeps freezing during checkout. Very annoying.",
        "Poor build quality. Plastic feels flimsy and screws fell out after one week.",
        "Misleading advertisement! Features listed on the box are completely absent.",
        "Customer service ignores emails and phone calls. Extremely untrustworthy seller.",
        "The restaurant was dirty, service was slow, and food gave me stomach trouble.",
        "Very unhappy with this service. Subscription renewed automatically despite cancellation.",
        "The fabric shrank significantly after just one wash. Low quality material.",
        "Sound quality is muffled and microphone barely picks up my voice during calls.",
        "Delivery took three weeks longer than promised and box was smashed."
    ]

    neutral_samples = [
        "The package arrived today as scheduled in standard cardboard packaging.",
        "The product dimensions are 10 inches by 5 inches and weighs approximately 2 pounds.",
        "Received the item yesterday. It matches the specifications listed on the website.",
        "The store opens at 9 AM and closes at 8 PM from Monday to Saturday.",
        "The flight departure is set for 2:30 PM from Gate 14.",
        "The software version 2.1 was released on Tuesday morning.",
        "The restaurant serves Italian and Mediterranean cuisine from 12 PM to 10 PM.",
        "The package contains one device, a charging cable, and a user manual.",
        "This product is available in three color options: black, white, and silver.",
        "The meeting has been rescheduled to Thursday at 3:00 PM in Conference Room B.",
        "The battery capacity is 4000 mAh and supports standard USB-C charging.",
        "The movie has a runtime of 125 minutes and is rated PG-13.",
        "The laptop features 16GB of RAM and 512GB SSD storage.",
        "Order #49201 was processed and shipped via standard ground transit.",
        "The warranty period lasts for 12 months from the date of purchase.",
        "The hotel is located two miles from the downtown city center.",
        "The update contains minor bug fixes and security patches.",
        "The temperature in the building is set to 72 degrees Fahrenheit.",
        "The user manual is available for download in PDF format on their website.",
        "The subscription plan auto-renews annually on the anniversary date."
    ]

    pos_templates = [
        "I am so glad I bought {prod}. It works {adj} and has {adj} value.",
        "Really {adj} experience with {prod}. The performance is truly {adj}.",
        "Five stars for {prod}! {adj} build, {adj} features, and {adj} delivery.",
        "{adj} quality! I am very happy with {prod} and will buy again.",
        "Exceeded expectations! {prod} is {adj} and worth every dollar.",
        "The team behind {prod} did an {adj} job. Very {adj} service!",
        "Loved everything about {prod}. Truly an {adj} product.",
        "Top-notch performance from {prod}. Highly {adj} and efficient.",
        "Superb experience! {prod} arrived {adj} and works like a charm.",
        "Impressed by how {adj} {prod} is. Outstanding overall quality!"
    ]

    neg_templates = [
        "Extremely {adj} experience with {prod}. It was a complete {adj} purchase.",
        "Do not buy {prod}! It is {adj}, broken, and useless.",
        "Very {adj} quality from {prod}. Failed after just a few days.",
        "Terrible customer service regarding {prod}. Agent was {adj} and unhelpful.",
        "Regret buying {prod}. The design is {adj} and performance is {adj}.",
        "{prod} was a huge disappointment. {adj} material and {adj} execution.",
        "The issue with {prod} was unresolved. Truly a {adj} product.",
        "Way too expensive for such a {adj} performance. {prod} is garbage.",
        "Frustrating and {adj}! {prod} stopped working immediately.",
        "Avoid {prod} at all costs. Extremely {adj} and poorly manufactured."
    ]

    neu_templates = [
        "The {prod} was delivered on {day} according to tracking.",
        "This model of {prod} comes with {detail} standard options.",
        "The specifications for {prod} indicate a weight of {detail}.",
        "Information regarding {prod} can be found on page {num} of the guide.",
        "Order containing {prod} was registered under reference number {num}.",
        "{prod} operates within the standard voltage range specified.",
        "The shipment of {prod} was processed at the central warehouse.",
        "The warranty coverage for {prod} is standard across all models.",
        "{prod} is manufactured in {detail} and distributed globally.",
        "The scheduled update for {prod} will occur next {day}."
    ]

    pos_adjs = ["fantastic", "amazing", "excellent", "wonderful", "outstanding", "superb", "brilliant", "great", "impressive", "top-grade"]
    neg_adjs = ["terrible", "awful", "horrible", "disappointing", "frustrating", "faulty", "dreadful", "miserable", "subpar", "defective"]
    products = ["this laptop", "the wireless earbuds", "this smart watch", "the kitchen blender", "the coffee machine", "this online course", "the gaming monitor", "this hotel booking", "the smartphone app", "this winter jacket"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    details = ["standard", "default", "certified", "regular", "specified"]
    nums = ["104", "208", "501", "882", "940"]

    data = []
    for text in positive_samples:
        data.append({"text": text, "sentiment": "positive"})
    for text in negative_samples:
        data.append({"text": text, "sentiment": "negative"})
    for text in neutral_samples:
        data.append({"text": text, "sentiment": "neutral"})

    np.random.seed(42)
    for _ in range(500):
        t = np.random.choice(pos_templates)
        text = t.format(prod=np.random.choice(products), adj=np.random.choice(pos_adjs))
        data.append({"text": text, "sentiment": "positive"})

    for _ in range(500):
        t = np.random.choice(neg_templates)
        text = t.format(prod=np.random.choice(products), adj=np.random.choice(neg_adjs))
        data.append({"text": text, "sentiment": "negative"})

    for _ in range(500):
        t = np.random.choice(neu_templates)
        text = t.format(prod=np.random.choice(products), day=np.random.choice(days), detail=np.random.choice(details), num=np.random.choice(nums))
        data.append({"text": text, "sentiment": "neutral"})

    edge_cases = [
        {"text": "Oh great, another delay on my morning flight. Just what I needed today!", "sentiment": "negative"},
        {"text": "The phone isn't bad at all, actually works surprisingly well.", "sentiment": "positive"},
        {"text": "It performs as expected, neither exceptional nor terrible.", "sentiment": "neutral"},
        {"text": "I was expecting a disaster, but it turned out to be quite decent.", "sentiment": "positive"},
        {"text": "Thanks for breaking my item and charging me double for shipping!", "sentiment": "negative"},
        {"text": "The color is blue and it functions within average parameters.", "sentiment": "neutral"},
        {"text": "Not the worst movie I've seen, but definitely far from good.", "sentiment": "negative"},
        {"text": "Unbelievable service, if by unbelievable you mean completely incompetent.", "sentiment": "negative"},
        {"text": "The user manual explains how to toggle the power switch on and off.", "sentiment": "neutral"},
        {"text": "I can't say I dislike it, it's pretty good for the price.", "sentiment": "positive"}
    ]
    data.extend(edge_cases)

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv('data/sentiment_dataset.csv', index=False)
    print(f"Dataset successfully generated with {len(df)} rows.")
    print("Class Distribution:")
    print(df['sentiment'].value_counts())

if __name__ == '__main__':
    generate_sentiment_dataset()
