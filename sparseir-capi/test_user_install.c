// test_user_install.c - ユーザーインストールのテスト
#include <stdio.h>
#include <sparseir_capi.h>

int main() {
    printf("Testing SparseIR C API (User Installation)\n");
    printf("==========================================\n");
    
    // カーネルを作成
    spir_kernel* kernel = spir_kernel_new_fermionic(1.0, 0.1);
    if (!kernel) {
        printf("❌ Failed to create kernel\n");
        return 1;
    }
    
    // ラムダ値を取得
    double lambda = spir_kernel_lambda(kernel);
    printf("✅ Kernel created successfully\n");
    printf("   Lambda: %.6f\n", lambda);
    
    // カーネルを解放
    spir_kernel_release(kernel);
    printf("✅ Kernel released\n");
    
    printf("\n🎉 User installation test completed successfully!\n");
    return 0;
}
