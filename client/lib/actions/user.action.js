"use server";

import User from "@/lib/models/user.model";
import { connect } from "@/lib/mongo.db";

export async function createUser(user) {
  try {
    await connect();

    const newUser = await User.create(user);

    return JSON.parse(JSON.stringify(newUser));
  } catch (error) {
    console.log(error);
  }
}

export async function updateUser(userId, updateData) {
  try {
    await connect();

    const updatedUser = await User.findByIdAndUpdate(
      userId,
      { $set: updateData },
      { new: true, runValidators: true }
    );

    if (!updatedUser) {
      throw new Error("User not found");
    }

    return JSON.parse(JSON.stringify(updatedUser));
  } catch (error) {
    console.error("Error updating user:", error);
    throw error;
  }
}

export async function getUserDetails(userId) {
  try {
    await connect();

    const newUser = await User.findOne({ _id: userId });

    if (!newUser) {
      throw new Error("User not found");
    }

    return JSON.parse(JSON.stringify(newUser));
  } catch (error) {
    console.error("Error updating user:", error);
    throw error;
  }
}
