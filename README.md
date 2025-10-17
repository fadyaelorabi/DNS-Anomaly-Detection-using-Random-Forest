# Detecting DNS Anomalies with Random Forest 🌳

Here's a step-by-step look at how we built a machine learning model to spot suspicious activity in DNS traffic. Our journey involved creating our own data, training a smart algorithm, and fine-tuning it to perfection.

---

## 🧪 Step 1: Crafting Our Data - A Digital Sandbox

Before we could detect anomalies, we needed data to play with. So, we built our own custom dataset from the ground up.

### Simulating DNS Queries:
We used the **scapy** library to generate realistic DNS queries. We created a list of common domain names and had our script send requests to various DNS servers, mimicking real-world traffic.

### Throwing a Wrench in the Works:
To make things interesting, we intentionally made 40% of our queries "anomalous." We did this by tweaking key attributes to look suspicious:
- **Time-To-Live (TTL):** Set to be unusually high or low.
- **Packet Size:** Made abnormally large or small.
- **Response Time:** Introduced artificial delays.
- **Response Codes:** Forced errors like SERVFAIL (server failure) or NXDOMAIN (domain doesn't exist).

### Playing Detective - Feature Extraction:
For every query and response, we collected crucial clues (features) like TTL, transaction ID, response time, and the lengths of the query and response. We also engineered some higher-level features, like how often a domain was being queried or the ratio of error responses.

### The Final Product:
Our efforts resulted in a dataset of 30,000 DNS queries, a mix of normal and anomalous traffic. This file, **dns_data2.csv**, became the foundation for our model. It contained 18 features and one all-important **anomaly_label** column marking each query as either "Normal" or "Anomalous."

---

## 🧹 Step 2: Getting the Data Ready for Action

Raw data is messy. We had to clean it up to make it suitable for our machine learning model.

### Speaking the Machine's Language:
Models work with numbers, not text. We converted our target labels, "Normal" and "Anomalous," into 0 and 1, respectively. We did the same for all other text-based columns.

### Finding What Matters:
We generated a correlation matrix to see which of our features had the strongest relationship with the anomaly_label. We discovered that several columns (like IP addresses, port numbers, and geographic region) had very weak correlations. To reduce noise and simplify our model, we dropped them.

### Leveling the Playing Field:
Some features had values ranging in the thousands, while others were very small. To prevent the larger-scale features from dominating the model, we scaled all our data to fit within a range of 0 to 1.

---

## ✂️ Step 3: Splitting the Data

With our data clean and ready, we split it into two parts:
- **80% for the Training Set:** This is what the model would learn from.
- **20% for the Testing Set:** This was held back to test the model's performance on unseen data.

---

## 🤖 Step 4: Choosing Our Champion - The Random Forest

We chose the **Random Forest** algorithm for this task. It's essentially a team of decision trees that vote on the final outcome. It's great for this kind of problem because:
- It's fantastic at finding complex patterns.
- It handles large amounts of data well.
- It's naturally resistant to overfitting.

Our initial model training, however, hit a common snag: overfitting. The model was basically memorizing the training data's answers instead of learning how to think for itself.

---

## 💥 Step 5: A Clever Trick - Fighting Overfitting with Noise

To combat this, we did something that sounds a bit crazy: we intentionally introduced more chaos. We flipped the labels on 4% of our data, randomly changing some "Normal" queries to "Anomalous" and vice-versa. This forces the model to learn the deeper, more general patterns instead of just memorizing the training examples.

---

## ⚙️ Step 6: Fine-Tuning the Engine

To find the absolute best version of our model, we used **GridSearchCV**. This tool is like a master chef trying out every possible combination of ingredients to find the perfect recipe. We had it test 27 different combinations of the following settings:
- **n_estimators:** [100, 500, 1000] (How many trees in the forest?)
- **max_depth:** [10, 20, 30] (How deep can each tree grow?)
- **min_samples_split:** [2, 10, 20] (How many samples are needed to split a branch?)

To make sure the results were reliable, we used **5-fold cross-validation**. This involves splitting the data into 5 parts, training on 4, and testing on the 5th, rotating each time. It's like giving the model five different pop quizzes to ensure it really knows its stuff.

---

## 🏆 Step 7: The Final Verdict

Finally, we took the champion model from our tuning process and evaluated it one last time on our noisy test data. This final test confirmed that our model could not only detect anomalies accurately but was also robust enough to handle the imperfections found in real-world data.
