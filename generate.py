from scapy.all import IP, UDP, DNS, DNSQR, sr1   #  library used to create and send DNS queries and capture responses.
import pandas as pd
import time
import random

domains_to_query = [ # l domains el bs2l 3nha w mstnya response 
    "google.com", "example.com", "openai.com", "nonexistent.example",
    "test.example.org", "rare.tld", "subdomain.google.com", "my.site.info"
]
dns_servers = ["8.8.8.8", "8.8.4.4", "1.1.1.1"] # el servers l hb3tlha l requests (google w cloudfare)
dns_regions = {# dictionary mapping dns server ips to their geographical regions hst3mlha f el feature extraction
    "8.8.8.8": "US",
    "8.8.4.4": "US",
    "1.1.1.1": "Global"
}
# List to store extracted features and labels
data = []

# Simulate DNS traffic and extract features for 30,000 queries
for _ in range(30000): # kol iteration 3obara 3n one request b el response bt3hta
    # b5leh kol mra y5tar random domain ys2l 3no w random server yb3tlo 
    domain_to_query = random.choice(domains_to_query)
    dns_server_ip = random.choice(dns_servers)
    # create dns query packet
    # b3ml create l ip packet ray7a l el dst l ana e5trto random fo2
    # b7dd en l packet htst3ml UPD protocol w da asln l dns sh8al byh 3la port 53 
    # then DNS() b3ml build l el dns query rd=1 y3ny 3yzaha recurive (mstnya mno ygebly final answer). qd b7dd l domain l hs2l 3leh
    query = IP(dst=dns_server_ip) / UDP(dport=53) / DNS(rd=1, qd=DNSQR(qname=domain_to_query))
    print(f"Sending DNS query for {domain_to_query} to {dns_server_ip}...")  # Message indicating the request
    start_time = time.time()
    response = sr1(query, verbose=0, timeout=2) # sr1 dy btb3t l packet w tsna single response
    end_time = time.time()
    
    # Check if response is received and print the result
    if response:
        print(f"Response received for {domain_to_query} from {dns_server_ip}.")  # Success message
    else:
        print(f"No response received for {domain_to_query} from {dns_server_ip}.")  # Failure message
    
    # Default feature values
    # el ttl value h5tarha random lw mafesh response 8er kda hakhod l ttl bta3t l response
    ttl = random.choice([10, 64, 3600]) if not response else response[IP].ttl
    transaction_id = random.randint(0, 65535) if not response else response[DNS].id # hena nfs l 7aga 
    rcode = random.choice(["No Error", "SERVFAIL", "NXDOMAIN", "REFUSED"]) if not response else \
        ["No Error", "Format Error", "Server Failure", "NXDOMAIN", "Not Implemented", "Refused"][response[DNS].rcode]
    anomaly_label = "Normal" # start with normal
    query_length = len(domain_to_query)
    time_taken = end_time - start_time  #  RTT measured between the sending and receiving of the DNS query.
    packet_size = len(query) if not response else len(response) 
    src_port = random.randint(1024, 65535) if not response else response[UDP].sport
    dst_port = 53
    
    # introduce artificial anomalies
    if random.random() < 0.4:  # random value between 0.0 & 1.0
        anomaly_label = "Anomalous" # change label
        ttl = random.choice([1, 50000])  # Very low or very high TTL
        rcode = random.choice(["SERVFAIL", "NXDOMAIN", "REFUSED"])  # Common error codes
        transaction_id = random.randint(0, 65535)
        time_taken = random.uniform(2.0, 5.0)  # Delayed responses
        packet_size = random.randint(1, 512)  # Abnormal packet size
        query_length = random.randint(1, 255)  # Abnormal query lengths
    else: # no anomalies yb2a normal query and response 
        # adding  small, random changes or variations 3shan tb2a k2nha real-world DNS traffic. (helps the model generalize better when detecting anomalies)
        ttl = ttl + random.randint(-5, 5)  # Small fluctuation in TTL (±5 seconds)
        packet_size = packet_size + random.randint(-10, 10)  # Small fluctuation in packet size
        time_taken = time_taken + random.uniform(-0.1, 0.1)  # Small fluctuation in response time
    
    #(count how often each domain is queried)
    query_volume = data.count(domain_to_query)  # how many times the same domain has been queried in the dataset
    
    # assigns a geographic region to the DNS server being queried, l dns_regions dy l dic bakhod l input dns_server_ip da k2no l key l byro7 ydwr byh f l dic 3la value mo3yna
    geo_region = dns_regions.get(dns_server_ip)
    
    # Calculates the ratio of "error" response codes (NXDOMAIN or SERVFAIL) to "no error" response codes (No Error) in the dataset.
    response_code_ratio = sum(1 for row in data if row["rcode"] in ["NXDOMAIN", "SERVFAIL"]) / \
                          max(sum(1 for row in data if row["rcode"] == "No Error"), 1)
    
    # Prepare feature set
    features = {
        "src_ip": "unknown" if not response else response[IP].src,
        "dst_ip": dns_server_ip,
        "query_name": domain_to_query,
        "transaction_id": transaction_id,
        "rcode": rcode,
        "ttl": ttl,
        "packet_size": packet_size,
        "time_taken": time_taken,
        "query_length": query_length,
        "response_length": packet_size if response else 0,
        "src_port": src_port,
        "dst_port": dst_port,
        "query_type": random.choice(["A", "AAAA", "MX", "TXT"]),  # Keep as categorical
        "anomaly_label": anomaly_label,  # Keep as categorical
        "entropy": len(set(str(transaction_id))),  # Old entropy calculation
        "response_code_category": "Error" if rcode != "No Error" else "No Error",
        "query_volume": query_volume,  # New feature: Query volume
        "geo_region": geo_region,  # New feature: Geographic region of DNS server
        "response_code_ratio": response_code_ratio  # New feature: Response code ratio
    }
    data.append(features)
# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_file = "dns_data2.csv"
df.to_csv(output_file, index=False)

print(f"Dataset saved to {output_file}")
