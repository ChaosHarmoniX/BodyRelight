#version 330

uniform uint analytic;

uniform uint hasNormalMap;
uniform uint hasAlbedoMap;

uniform sampler2D AlbedoMap;
uniform sampler2D NormalMap;

in VertexData {
    vec3 Position;
    vec3 Depth;
    vec3 ModelNormal;
    vec2 Texcoord;
    vec3 Tangent;
    vec3 Bitangent;
} VertexIn;

layout (location = 0) out vec4 FragColor;
layout (location = 1) out vec4 FragNormal;
layout (location = 2) out vec4 FragPosition;
layout (location = 3) out vec4 FragAlbedo;
layout (location = 4) out vec4 FragShading;

vec4 gammaCorrection(vec4 vec, float g)
{
    return vec4(pow(vec.x, 1.0/g), pow(vec.y, 1.0/g), pow(vec.z, 1.0/g), vec.w);
}

vec3 gammaCorrection(vec3 vec, float g)
{
    return vec3(pow(vec.x, 1.0/g), pow(vec.y, 1.0/g), pow(vec.z, 1.0/g));
}

// void evaluateH(vec3 n, out float H[9])
// {
//     float c1 = 0.429043, c2 = 0.511664,
//         c3 = 0.743125, c4 = 0.886227, c5 = 0.247708;

//     H[0] = c4;
//     H[1] = 2.0 * c2 * n[1];
//     H[2] = 2.0 * c2 * n[2];
//     H[3] = 2.0 * c2 * n[0];
//     H[4] = 2.0 * c1 * n[0] * n[1];
//     H[5] = 2.0 * c1 * n[1] * n[2];
//     H[6] = c3 * n[2] * n[2] - c5;
//     H[7] = 2.0 * c1 * n[2] * n[0];
//     H[8] = c1 * (n[0] * n[0] - n[1] * n[1]);
// }

// vec3 evaluateLightingModel(vec3 normal)
// {
//     float H[9];
//     evaluateH(normal, H);
//     vec3 res = vec3(0.0);
//     for (int i = 0; i < 9; i++) {
//         res += H[i] * SHCoeffs[i];
//     }
//     return res;
// }

// // nC: coarse geometry normal, nH: fine normal from normal map
// vec3 evaluateLightingModelHybrid(vec3 nC, vec3 nH, mat3 prt)
// {
//     float HC[9], HH[9];
//     evaluateH(nC, HC);
//     evaluateH(nH, HH);
    
//     vec3 res = vec3(0.0);
//     vec3 shadow = vec3(0.0);
//     vec3 unshadow = vec3(0.0);
//     for(int i = 0; i < 3; ++i){
//         for(int j = 0; j < 3; ++j){
//             int id = i*3+j;
//             res += HH[id]* SHCoeffs[id];
//             shadow += prt[i][j] * SHCoeffs[id];
//             unshadow += HC[id] * SHCoeffs[id];
//         }
//     }
//     vec3 ratio = clamp(shadow/unshadow,0.0,1.0);
//     res = ratio * res;

//     return res;
// }

// vec3 evaluateLightingModelPRT(mat3 prt)
// {
//     vec3 res = vec3(0.0);
//     for(int i = 0; i < 3; ++i){
//         for(int j = 0; j < 3; ++j){
//             res += prt[i][j] * SHCoeffs[i*3+j];
//         }
//     }

//     return res;
// }

void main()
{
    vec2 uv = VertexIn.Texcoord;
    vec3 nC = normalize(VertexIn.ModelNormal);

    FragShading = vec4(nC, 1.0f);


    FragShading = gammaCorrection(FragShading, 2.2);
    // FragColor = clamp(FragAlbedo * FragShading, 0.0, 1.0);
    FragColor = texture(AlbedoMap, uv);
    FragNormal = vec4(0.5*(nC+vec3(1.0)), 1.0);
    FragPosition = vec4(VertexIn.Position,VertexIn.Depth.x);
    FragShading = vec4(clamp(0.5*FragShading.xyz, 0.0, 1.0),1.0);



}