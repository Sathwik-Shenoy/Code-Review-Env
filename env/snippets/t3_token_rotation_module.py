from datetime import datetime, timedelta


def should_rotate(last_rotated_at, now=None):
    now = now or datetime.utcnow()
    age = now - last_rotated_at
    return age > timedelta(days=30)


def rotate_token(user_record, signer):
    if not should_rotate(user_record["last_rotated_at"]):
        return user_record["token"]

    token = signer.sign(str(user_record["id"]))
    user_record["token"] = token
    return token
