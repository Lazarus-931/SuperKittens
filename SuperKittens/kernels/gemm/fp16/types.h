//
//  types.metal
//  SuperKittens
//
//  Created by Alazar Manakelew on 4/1/26.
//





                                                                                                                                                                                    
// BM=32, BN=32, BK=16 with WM=2, WN=2
struct fp16_1_config {
    enum : uint {
        BM = 32,
        BN = 32,
        BK = 16,
        WM = 2,
        WN = 2
    };
};



// BM=64, BN=64, BK=16 with WM=2, WN=2
struct fp16_2_config {
    enum : uint {
        BM=64,
        BN=64,
        BK=16,
        WM=2,
        WN=2
    };
};



