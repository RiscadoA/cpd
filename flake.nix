# Flake used for development with nix
{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-23.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = inputs:
    inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import inputs.nixpkgs { inherit system; };
      in
      {
        devShell = pkgs.mkShell {
          hardeningDisable = [ "fortify" ];

          packages = with pkgs; [
            gcc
            clang-tools
            linuxKernel.packages.linux_6_1.perf
            bc
            openmpi
          ];
        };
      });
}
