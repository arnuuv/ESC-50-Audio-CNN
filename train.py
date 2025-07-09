import sys
import modal
app = modal.App("audio-cnn")

image = (modal.Image.debain_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget", "unzip"])
         )


@app.function()
def f(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)

    return i * i


@app.local_entrypoint()
def main():
    # run the function locally
    print(f.local(10))

    # run the function remotely on Modal
    print(f.remote(10))
