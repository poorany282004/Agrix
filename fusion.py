import random, datetime

def simulate_sensors():
    sensors = {
        'temperature': round(random.uniform(15.0, 35.0), 2),
        'humidity': round(random.uniform(30.0, 95.0), 2),
        'soil_moisture': round(random.uniform(5.0, 60.0), 2),
        'ph': round(random.uniform(4.5, 8.5), 2)
    }
    return sensors

def env_risk_from_sensors(sensors):
    temp = sensors['temperature']
    hum = sensors['humidity']
    sm = sensors['soil_moisture']
    ph = sensors['ph']

    temp_risk = max(0.0, 1.0 - abs(25.0 - temp) / 20.0)
    hum_risk = min(max((hum - 40.0) / 60.0, 0.0), 1.0)
    if sm < 10:
        sm_risk = 0.2
    elif sm < 30:
        sm_risk = 0.6
    else:
        sm_risk = 0.8
    ph_risk = min(max(abs(6.5 - ph) / 3.0, 0.0), 1.0)
    env_score = 0.4 * temp_risk + 0.35 * hum_risk + 0.15 * sm_risk + 0.1 * ph_risk
    return min(max(env_score, 0.0), 1.0)

def compute_fusion_and_stage(img_pred, sensors, alpha=0.7):
    img_prob = float(img_pred.get('prob', 0.0))
    env_risk = env_risk_from_sensors(sensors)
    fusion_score = alpha * img_prob + (1.0 - alpha) * env_risk
    if fusion_score >= 0.70:
        stage = 'FULLY_INFECTED'
    elif fusion_score >= 0.40:
        stage = 'PARTIALLY_INFECTED'
    else:
        stage = 'HEALTHY'
    return {
        'img_prob': round(img_prob, 4),
        'env_risk': round(env_risk, 4),
        'fusion_score': round(fusion_score, 4),
        'stage': stage,
        'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
    }
