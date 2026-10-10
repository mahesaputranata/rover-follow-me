import socket, pickle, struct, time

def get_frame_from_service(port: int = 6001, retries: int = 5, delay: float = 0.5):
    last_error = None
    for _ in range(retries):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect(("127.0.0.1", port))
            header = s.recv(4)
            size = struct.unpack(">I", header)[0]
            data = b""
            while len(data) < size:
                data += s.recv(size - len(data))
            s.close()
            return pickle.loads(data)
        except (ConnectionRefusedError, ConnectionResetError) as e:
            last_error = e
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to camera service after {retries} attempts: {last_error}")