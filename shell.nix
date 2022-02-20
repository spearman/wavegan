# $ nix-shell
# $ pip install gast==0.3.3
# $ pip install tensorflow-cpu python-language-server matplotlib
# $ pip install scipy==1.0.0
# $ pip install librosa==0.6.2
# $ ./train.sh

with import <nixpkgs> {};
mkShell {
  buildInputs = [
    (python39.withPackages (p: with p; [
      pip
      #python-language-server  # doesn't work with python 3.9; install with pip
    ]))
  ];
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    #alias pip="PIP_PREFIX='$(pwd)/_build/pip_packages' TMPDIR='$(pwd)' /pip"
    export PIP_PREFIX="$(pwd)/_build/pip_packages"
    export TMPDIR="$(pwd)"
    export PYTHONPATH="$(pwd)/_build/pip_packages/lib/python3.9/site-packages:$PYTHONPATH"
    #export PATH="$(pwd)/_build/pip_packages/bin:$PATH"
    unset SOURCE_DATE_EPOCH
  '';
}
