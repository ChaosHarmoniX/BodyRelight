#version 330

layout (location = 0) in vec3 a_Position;
layout (location = 1) in vec3 a_Normal;
layout (location = 2) in vec2 a_TextureCoord;
layout (location = 3) in vec3 a_Tangent;
layout (location = 4) in vec3 a_Bitangent;

out VertexData {
    vec3 Position;
    vec3 Depth;
    vec3 ModelNormal;
    vec2 Texcoord;
    vec3 Tangent;
    vec3 Bitangent;
} VertexOut;

uniform mat3 RotMat;
uniform mat4 NormMat;
uniform mat4 ModelMat;
uniform mat4 PerspMat;


void main()
{
    // normalization
    vec3 pos = (NormMat * vec4(a_Position,1.0)).xyz;

    mat3 R = mat3(ModelMat) * RotMat;
    VertexOut.ModelNormal = (R * a_Normal);
    VertexOut.Position = R * pos;
    VertexOut.Texcoord = a_TextureCoord;
    VertexOut.Tangent = (R * a_Tangent);
    VertexOut.Bitangent = (R * a_Bitangent);

    gl_Position = PerspMat * ModelMat * vec4(RotMat * pos, 1.0);
    
}
