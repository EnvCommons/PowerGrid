from openreward.environments import Server

from powergrid import PowerGridEnvironment

if __name__ == "__main__":
    server = Server([PowerGridEnvironment])
    server.run(port=8080)
