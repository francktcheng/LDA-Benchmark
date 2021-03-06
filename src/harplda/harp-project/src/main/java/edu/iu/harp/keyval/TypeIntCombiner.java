/*
 * Copyright 2013-2017 Indiana University
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package edu.iu.harp.keyval;

/*******************************************************
 * Combiner for merging int-type Values
 ******************************************************/
public class TypeIntCombiner {

  /**
   * Merge two int-type values
   * 
   * @param curVal
   *          the current value
   * @param newVal
   *          the new value
   * @return the merged value
   */
  public int combine(int curVal, int newVal) {
    return curVal + newVal;
  }
}
