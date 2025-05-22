import socket
import threading
import time

# this is the function that should be called by outside scripts
# this will be called to asynchronously create a connection and receive data from the client.
def connect_async(ip_address, port, onDataReceived):
    threading.Thread(target=connect, args=(ip_address, port, onDataReceived)).start()

# same as connect_async, but has no threading
def connect(ip_address, port, onDataReceived):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip_address, port))
    server.listen()
    print(f"Server listening on port {port}...")
    while True:
        conn, addr = server.accept()
        threading.Thread(target=handle_client, args=(conn, addr, onDataReceived)).start()

# called by connect(). use to receive data
def handle_client(conn, addr, onDataReceived):
    try:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            onDataReceived(data);
        conn.close()
    except socket.error:
        print("socket: lost connection")
    except ConnectionResetError:
        print("socket: lost connection")
    finally:
        conn.close();

def m_cb(data):
    print(f"received data: {data}")

if __name__ == "__main":
    connect_async("192.168.3.205", 5000, m_cb);
    while True:
        time.sleep(3000);
