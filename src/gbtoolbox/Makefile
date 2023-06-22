cc=gcc
INC_DIR=/usr/local/include
LD_DIR=/usr/local/lib

main:
	cc -O3 -shared -I $(INC_DIR) -L $(LD_DIR) -ffast-math -o libft.so -fPIC ft_tools.c -lm
	cc -O3 -shared -I $(INC_DIR) -L $(LD_DIR) -ffast-math -o libztw.so -fPIC gen_ztw.c -lm
clean:
	rm -rf libft.so libztw.so
install:
	sudo mkdir -p $(LD_DIR)/gbtoolbox
	sudo cp lib* $(LD_DIR)/gbtoolbox
uninstall:
	sudo rm -rf $(LD_DIR)/gbtoolbox


